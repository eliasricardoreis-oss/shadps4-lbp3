// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <ranges>
#include <thread>

#include "common/guest_time_stall.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/performance_telemetry.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/copy_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

struct PipelineCache::AsyncCompiler {
    std::mutex mutex;
    std::condition_variable_any job_cv;
    std::condition_variable_any idle_cv;
    std::condition_variable_any completion_cv;
    std::deque<std::function<void()>> jobs;
    std::deque<std::function<void()>> completions;
    size_t active_jobs{};
    Shader::Pools pools;
    vk::UniquePipelineCache pipeline_cache;
    std::jthread worker;
};

namespace {

struct AsyncShaderCompileRequest {
    Shader::Stage stage{};
    Shader::LogicalStage logical_stage{};
    Shader::RuntimeInfo runtime_info{};
    Shader::Backend::Bindings start_binding{};
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::vector<u32> code;
    std::vector<u32> geometry_copy_code;
    u64 hash{};
    size_t permutation_index{};
    bool is_base_program{};
    Shader::StageSpecialization specialization{};
    std::unique_ptr<Shader::Info> info;
    std::vector<u32> spirv;
    std::vector<u32> patch;
    vk::ShaderModule module{};
    bool is_patched{};
    std::string error;
};

struct AsyncGraphicsPipelineRequest {
    GraphicsPipelineKey key{};
    u64 hash{};
    std::array<const Shader::Info*, MaxShaderStages> live_infos{};
    std::array<const Shader::Info*, MaxShaderStages> snapshot_info_ptrs{};
    std::array<std::unique_ptr<Shader::Info>, MaxShaderStages> snapshot_infos{};
    std::array<std::array<u32, Shader::ShaderParams::NumShaderUserData>, MaxShaderStages>
        snapshot_user_data{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<std::vector<u32>, MaxShaderStages> geometry_copy_code{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipeline::SerializationSupport serialization{};
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::string error;
};

bool UseAsyncGraphicsCompilation(const Instance& instance) {
    const char* setting = std::getenv("SHADPS4_ASYNC_GRAPHICS_COMPILATION");
    if (setting != nullptr) {
        return !(setting[0] == '0' && setting[1] == '\0');
    }
    return instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp;
}

void SnapshotGeometryCopyShader(Shader::RuntimeInfo& runtime_info, std::vector<u32>& storage) {
    if (runtime_info.stage != Shader::Stage::Geometry || runtime_info.gs_info.vs_copy.empty()) {
        return;
    }
    storage.assign(runtime_info.gs_info.vs_copy.begin(), runtime_info.gs_info.vs_copy.end());
    runtime_info.gs_info.vs_copy = storage;
}

} // Anonymous namespace

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            (regs.primitive_type == AmdGpu::PrimitiveType::QuadList &&
             !graphics_key.expand_quad_list);
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;

        // Scenario G can program MAX_VERT_OUT and VERT_ITEMSIZE with allocation values that are
        // larger/smaller than the actual sparse GSVS layout consumed by the copy shader. The
        // recompiler uses the copy shader to eliminate ring accesses, but its SPIR-V entry point
        // still takes OutputVertices from this RuntimeInfo. Correct both values here so the native
        // geometry-stage interface and the eliminated ring-offset mapping describe the same
        // layout.
        const auto copy_data = Shader::ParseCopyShader(gs_info.vs_copy);
        if (copy_data.output_vertices && copy_data.output_vertices < gs_info.output_vertices &&
            gs_info.mode == AmdGpu::GsScenario::ScenarioG) {
            LOG_TRACE(Render_Vulkan, "Correcting GS MAX_VERT_OUT {} to copy-shader vertex count {}",
                      gs_info.output_vertices, copy_data.output_vertices);
            gs_info.output_vertices = copy_data.output_vertices;
        }
        if (!copy_data.attr_map.empty() && copy_data.output_vertices) {
            // Copy-shader offsets advance by 64 bytes per GSVS component per output vertex.
            // Use the final occupied slot rather than num_comps: sparse exports still consume
            // their holes in the component-major ring layout.
            const u32 component_stride = copy_data.output_vertices * 64u;
            const u32 last_component_offset = copy_data.attr_map.rbegin()->first;
            if (last_component_offset % component_stride == 0) {
                const u32 component_span = last_component_offset / component_stride + 1u;
                if (component_span > gs_info.out_vertex_data_size) {
                    LOG_TRACE(Render_Vulkan,
                              "Correcting GS VERT_ITEMSIZE {} to copy-shader component span {} "
                              "({} populated components)",
                              gs_info.out_vertex_data_size, component_span, copy_data.num_comps);
                    gs_info.out_vertex_data_size = component_span;
                }
            } else {
                LOG_WARNING(Render_Vulkan,
                            "Copy-shader final offset {} is not aligned to component stride {}",
                            last_component_offset, component_stride);
            }
        }
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        info.fs_info.clip_distance_emulation =
            regs.vs_output_control.clip_distance_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local)) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
    // Register the SRT fault handler before the async compiler can generate or execute walkers.
    // Registering into the global signal dispatcher from a worker would race signal dispatch.
    Shader::InitializeSrtWalker();

    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
        .supports_shader_output_layer = instance_.IsShaderOutputLayerSupported(),
    };
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    // Preloaded pipelines must populate the same Vulkan cache used for first-use pipelines.
    // WarmUp constructs actual host pipelines, so running it before creating this cache passes
    // VK_NULL_HANDLE to all of them and throws away potential driver-side reuse for later state
    // variants.
    WarmUp();

    if (UseAsyncGraphicsCompilation(instance) && !EmulatorSettings.IsShaderCollect()) {
        async_compiler = std::make_unique<AsyncCompiler>();
        auto [async_cache_result, async_cache] = instance.GetDevice().createPipelineCacheUnique({});
        ASSERT_MSG(async_cache_result == vk::Result::eSuccess,
                   "Failed to create async pipeline cache: {}", vk::to_string(async_cache_result));
        async_compiler->pipeline_cache = std::move(async_cache);
        async_compiler->worker =
            std::jthread{[this](std::stop_token stop_token) { AsyncCompilerThread(stop_token); }};
        LOG_INFO(Render_Vulkan, "Asynchronous graphics shader/pipeline compilation enabled; "
                                "compute compilation remains synchronous");
    } else if (UseAsyncGraphicsCompilation(instance)) {
        LOG_WARNING(Render_Vulkan,
                    "Asynchronous graphics compilation disabled while shader collection is "
                    "active");
    }
}

PipelineCache::~PipelineCache() {
    // Flush every queued shader/pipeline record while the cache object is still alive. Relying on
    // process-global destruction can stop the IO worker before the last gameplay discoveries reach
    // disk, which makes the next run compile those shaders again.
    Sync();
    StopAsyncCompiler();
}

void PipelineCache::QueueAsyncJob(std::function<void()> job) {
    ASSERT(async_compiler);
    {
        std::scoped_lock lock{async_compiler->mutex};
        async_compiler->jobs.emplace_back(std::move(job));
    }
    async_compiler->job_cv.notify_one();
}

void PipelineCache::QueueAsyncCompletion(std::function<void()> completion) {
    ASSERT(async_compiler);
    {
        std::scoped_lock lock{async_compiler->mutex};
        async_compiler->completions.emplace_back(std::move(completion));
    }
    async_compiler->completion_cv.notify_all();
}

void PipelineCache::DrainAsyncCompletions() {
    if (!async_compiler) {
        return;
    }

    std::deque<std::function<void()>> completions;
    {
        std::scoped_lock lock{async_compiler->mutex};
        completions.swap(async_compiler->completions);
    }
    for (auto& completion : completions) {
        completion();
    }
}

void PipelineCache::WaitForAsyncCompiler() {
    if (!async_compiler) {
        return;
    }

    {
        std::unique_lock lock{async_compiler->mutex};
        async_compiler->idle_cv.wait(lock, [this] {
            return async_compiler->jobs.empty() && async_compiler->active_jobs == 0;
        });
    }
    DrainAsyncCompletions();
}

void PipelineCache::WaitForAsyncGraphicsProgress() {
    if (!async_compiler) {
        return;
    }

    std::unique_lock lock{async_compiler->mutex};
    if (!async_compiler->completions.empty()) {
        return;
    }
    async_compiler->completion_cv.wait_for(lock, std::chrono::microseconds{500},
                                           [this] { return !async_compiler->completions.empty(); });
}

void PipelineCache::StopAsyncCompiler() {
    if (!async_compiler) {
        return;
    }

    async_compiler->worker.request_stop();
    async_compiler->job_cv.notify_all();
    if (async_compiler->worker.joinable()) {
        async_compiler->worker.join();
    }
    DrainAsyncCompletions();
    async_compiler.reset();
}

void PipelineCache::AsyncCompilerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("shadPS4:ShaderCompiler");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);

    while (!stop_token.stop_requested()) {
        std::function<void()> job;
        {
            std::unique_lock lock{async_compiler->mutex};
            if (!async_compiler->job_cv.wait(lock, stop_token,
                                             [this] { return !async_compiler->jobs.empty(); })) {
                break;
            }
            if (stop_token.stop_requested()) {
                break;
            }
            job = std::move(async_compiler->jobs.front());
            async_compiler->jobs.pop_front();
            ++async_compiler->active_jobs;
        }

        try {
            job();
        } catch (const std::exception& exception) {
            LOG_ERROR(Render_Vulkan, "Unhandled async compiler exception: {}", exception.what());
        } catch (...) {
            LOG_ERROR(Render_Vulkan, "Unhandled unknown async compiler exception");
        }

        {
            std::scoped_lock lock{async_compiler->mutex};
            --async_compiler->active_jobs;
            if (async_compiler->jobs.empty() && async_compiler->active_jobs == 0) {
                async_compiler->idle_cv.notify_all();
            }
        }
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(bool expand_quad_list) {
    graphics_compilation_pending = false;
    DrainAsyncCompletions();
    if (!RefreshGraphicsKey(expand_quad_list)) {
        return nullptr;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        if (async_compiler) {
            auto request = std::make_shared<AsyncGraphicsPipelineRequest>();
            request->key = graphics_key;
            request->hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
            request->live_infos = infos;
            request->runtime_infos = runtime_infos;
            for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
                SnapshotGeometryCopyShader(request->runtime_infos[stage],
                                           request->geometry_copy_code[stage]);
            }
            request->modules = modules;
            request->fetch_shader = fetch_shader;

            for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage] == nullptr) {
                    continue;
                }
                request->snapshot_infos[stage] = std::make_unique<Shader::Info>(*infos[stage]);
                auto& user_data = request->snapshot_user_data[stage];
                const auto source_user_data = infos[stage]->user_data;
                const size_t copy_count = std::min(user_data.size(), source_user_data.size());
                std::ranges::copy_n(source_user_data.begin(), copy_count, user_data.begin());
                request->snapshot_infos[stage]->user_data = user_data;
                request->snapshot_info_ptrs[stage] = request->snapshot_infos[stage].get();
            }

            QueueAsyncJob([this, request] {
                Core::PerfTelemetry::Increment(
                    Core::PerfTelemetry::Counter::GraphicsPipelineCompiles);
                Core::PerfTelemetry::ScopedTimer telemetry_timer{
                    Core::PerfTelemetry::TimeMetric::GraphicsPipelineCompile};
                DebugState.BeginShaderCompile(DebugStateType::ShaderCompileKind::GraphicsPipeline);
                SCOPE_EXIT {
                    DebugState.EndShaderCompile();
                };
                LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x} asynchronously",
                         request->hash);
                try {
                    request->pipeline = std::make_unique<GraphicsPipeline>(
                        instance, scheduler, desc_heap, profile, request->key,
                        *async_compiler->pipeline_cache, request->snapshot_info_ptrs,
                        request->runtime_infos, request->fetch_shader, request->modules,
                        request->serialization, false);
                } catch (const std::exception& exception) {
                    request->error = exception.what();
                } catch (...) {
                    request->error = "unknown exception";
                }

                QueueAsyncCompletion([this, request] {
                    const auto pipeline_it = graphics_pipelines.find(request->key);
                    if (pipeline_it == graphics_pipelines.end() || pipeline_it->second) {
                        return;
                    }
                    if (!request->pipeline) {
                        LOG_ERROR(Render_Vulkan, "Async graphics pipeline {:#x} failed: {}",
                                  request->hash, request->error);
                        graphics_pipelines.erase(pipeline_it);
                        return;
                    }

                    request->pipeline->RebindStageInfos(request->live_infos);
                    pipeline_it.value() = std::move(request->pipeline);
                    RegisterPipelineData(request->key, request->hash, request->serialization);
                    ++num_new_pipelines;

                    if (EmulatorSettings.IsShaderCollect()) {
                        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
                            if (request->live_infos[stage]) {
                                module_related_pipelines[request->modules[stage]].emplace_back(
                                    request->key);
                            }
                        }
                    }
                });
            });
            fetch_shader.reset();
            graphics_compilation_pending = true;
            return nullptr;
        }

        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::GraphicsPipelineCompiles);
        Core::PerfTelemetry::ScopedTimer telemetry_timer{
            Core::PerfTelemetry::TimeMetric::GraphicsPipelineCompile};
        DebugState.BeginShaderCompile(DebugStateType::ShaderCompileKind::GraphicsPipeline);
        SCOPE_EXIT {
            DebugState.EndShaderCompile();
        };
        const Common::GuestTimeStallScope guest_time_stall;
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
    }
    if (!it->second) {
        graphics_compilation_pending = true;
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    DrainAsyncCompletions();
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::ComputePipelineCompiles);
        Core::PerfTelemetry::ScopedTimer telemetry_timer{
            Core::PerfTelemetry::TimeMetric::ComputePipelineCompile};
        DebugState.BeginShaderCompile(DebugStateType::ShaderCompileKind::ComputePipeline);
        SCOPE_EXIT {
            DebugState.EndShaderCompile();
        };
        const Common::GuestTimeStallScope guest_time_stall;
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey(bool expand_quad_list) {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.expand_quad_list = expand_quad_list;
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

    enum class BindStageResult {
        Disabled,
        Ready,
        Pending,
    };

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in,
                                Shader::LogicalStage stage_out) -> BindStageResult {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return BindStageResult::Disabled;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return BindStageResult::Disabled;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        const auto program = GetProgram(stage_in, stage_out, params, binding);
        if (!program) {
            infos[stage_out_idx] = nullptr;
            modules[stage_out_idx] = nullptr;
            return BindStageResult::Pending;
        }
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) = *program;
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return BindStageResult::Ready;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    const auto result = bind_stage(Stage::Fragment, LogicalStage::Fragment);
    if (result == BindStageResult::Pending) {
        return false;
    }
    if (result == BindStageResult::Disabled && regs.vs_output_control.clip_distance_enable &&
        profile.needs_clip_distance_emulation) {
        // TODO: need to implement a discard only fallback shader
        LOG_WARNING(Render_Vulkan,
                    "Clip distance emulation is ineffective due to absense of fragment shader");
    }

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (bind_stage(Stage::Export, LogicalStage::Vertex) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Geometry, LogicalStage::Geometry) != BindStageResult::Ready) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (bind_stage(Stage::Hull, LogicalStage::TessellationControl) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Vertex, LogicalStage::TessellationEval) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Local, LogicalStage::Vertex) != BindStageResult::Ready) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (bind_stage(Stage::Hull, LogicalStage::TessellationControl) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Export, LogicalStage::TessellationEval) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Local, LogicalStage::Vertex) != BindStageResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Geometry, LogicalStage::Geometry) != BindStageResult::Ready) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs: {
        const auto vertex_result = bind_stage(Stage::Vertex, LogicalStage::Vertex);
        if (vertex_result == BindStageResult::Pending) {
            return false;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    const auto program =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    if (!program) {
        return false;
    }
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) = *program;
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::GuestShaderCompiles);
    Core::PerfTelemetry::ScopedTimer telemetry_timer{
        Core::PerfTelemetry::TimeMetric::GuestShaderCompile};
    DebugState.BeginShaderCompile(DebugStateType::ShaderCompileKind::GuestShader);
    SCOPE_EXIT {
        DebugState.EndShaderCompile();
    };
    const Common::GuestTimeStallScope guest_time_stall;
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

std::optional<PipelineCache::Result> PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                               const Shader::ShaderParams& params,
                                                               Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(stage, l_stage);
    // LBP3's sprite-light normalize/tone-map vertex shader renders one 2D-array slice per draw.
    // This otherwise-generic fullscreen VS is reused by pipelines with other primitive topologies,
    // so writing Layer must be specialized on the two proven fragment shaders and their expanded
    // QuadList topology rather than on the VS hash alone.
    static constexpr u64 Lbp3SpriteLightVertexHash = 0xd44ad72f3a6cfdcaULL;
    static constexpr u64 Lbp3SpriteLightNormalizeHash = 0xd02859f9905c939eULL;
    static constexpr u64 Lbp3SpriteLightToneMapHash = 0xc79fdb84db57fd1eULL;
    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    const bool is_lbp3_sprite_light_fs =
        fs_info && (fs_info->pgm_hash == Lbp3SpriteLightNormalizeHash ||
                    fs_info->pgm_hash == Lbp3SpriteLightToneMapHash);
    if (stage == Stage::Vertex && l_stage == LogicalStage::Vertex &&
        params.hash == Lbp3SpriteLightVertexHash && is_lbp3_sprite_light_fs &&
        graphics_key.prim_type == AmdGpu::PrimitiveType::QuadList &&
        graphics_key.expand_quad_list && profile.supports_shader_output_layer) {
        runtime_info.vs_info.force_host_layer_output = true;
    }

    const auto queue_async_shader = [&](bool is_base_program, size_t permutation_index,
                                        const Shader::StageSpecialization* specialization) {
        auto request = std::make_shared<AsyncShaderCompileRequest>();
        request->stage = stage;
        request->logical_stage = l_stage;
        request->runtime_info = runtime_info;
        SnapshotGeometryCopyShader(request->runtime_info, request->geometry_copy_code);
        request->start_binding = binding;
        request->hash = params.hash;
        request->permutation_index = permutation_index;
        request->is_base_program = is_base_program;
        std::ranges::copy(params.user_data, request->user_data.begin());
        request->code.assign(params.code.begin(), params.code.end());
        if (specialization) {
            request->specialization = *specialization;
        }

        const Shader::ShaderParams snapshot_params{
            .user_data = request->user_data,
            .code = request->code,
            .hash = request->hash,
        };
        request->info = std::make_unique<Shader::Info>(stage, l_stage, snapshot_params);

        QueueAsyncJob([this, request] {
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::GuestShaderCompiles);
            Core::PerfTelemetry::ScopedTimer telemetry_timer{
                Core::PerfTelemetry::TimeMetric::GuestShaderCompile};
            DebugState.BeginShaderCompile(DebugStateType::ShaderCompileKind::GuestShader);
            SCOPE_EXIT {
                DebugState.EndShaderCompile();
            };

            LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {} asynchronously", request->stage,
                     request->hash, request->permutation_index != 0 ? "(permutation)" : "");
            try {
                DumpShader(request->code, request->hash, request->stage, request->permutation_index,
                           "bin");

                auto worker_runtime_info = request->runtime_info;
                auto worker_binding = request->start_binding;
                const auto ir_program =
                    Shader::TranslateProgram(request->code, async_compiler->pools, *request->info,
                                             worker_runtime_info, profile);
                request->spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, worker_runtime_info,
                                                                   ir_program, worker_binding);
                DumpShader(request->spirv, request->hash, request->stage,
                           request->permutation_index, "spv");

                auto patch = GetShaderPatch(request->hash, request->stage,
                                            request->permutation_index, "spv");
                if (patch) {
                    request->patch = std::move(*patch);
                }
                request->is_patched = !request->patch.empty() && EmulatorSettings.IsPatchShaders();
                if (request->is_patched) {
                    LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", request->stage,
                             request->hash);
                    request->module = CompileSPV(request->patch, instance.GetDevice());
                } else {
                    request->module = CompileSPV(request->spirv, instance.GetDevice());
                }

                if (request->is_base_program) {
                    request->specialization = Shader::StageSpecialization(
                        *request->info, worker_runtime_info, profile, request->start_binding);
                }

                const auto name =
                    GetShaderName(request->stage, request->hash, request->permutation_index);
                Vulkan::SetObjectName(instance.GetDevice(), request->module, name);
            } catch (const std::exception& exception) {
                request->error = exception.what();
            } catch (...) {
                request->error = "unknown exception";
            }

            // The request-owned register/code snapshots go away after publication. Live values
            // are restored by GetProgram whenever the shader is selected for a draw.
            request->info->pgm_base = 0;
            request->info->user_data = {};

            QueueAsyncCompletion([this, request] {
                const u64 permutation_hash = HashCombine(request->hash, request->permutation_index);
                const auto destroy_request_module = [&] {
                    if (request->module) {
                        instance.GetDevice().destroyShaderModule(request->module);
                        request->module = nullptr;
                    }
                };

                if (request->is_base_program) {
                    async_pending_programs.erase(request->hash);
                    if (!request->error.empty() || !request->module) {
                        LOG_ERROR(Render_Vulkan, "Async {} shader {:#x} failed: {}", request->stage,
                                  request->hash, request->error);
                        destroy_request_module();
                        return;
                    }
                    if (program_cache.contains(request->hash)) {
                        destroy_request_module();
                        return;
                    }

                    auto program = std::make_unique<Program>();
                    program->info = std::move(*request->info);
                    request->specialization.info = &program->info;
                    RegisterShaderBinary(std::move(request->spirv), request->hash,
                                         request->permutation_index);
                    RegisterShaderMeta(program->info, request->specialization.fetch_shader_data,
                                       request->specialization, permutation_hash,
                                       request->permutation_index);
                    program->AddPermut(request->module, std::move(request->specialization));
                    request->module = nullptr;
                    program_cache[request->hash] = std::move(program);
                    return;
                }

                const auto program_it = program_cache.find(request->hash);
                if (program_it == program_cache.end() ||
                    request->permutation_index >= program_it->second->modules.size()) {
                    destroy_request_module();
                    return;
                }
                auto& program = *program_it->second;
                auto& module = program.modules[request->permutation_index];
                if (module.module) {
                    destroy_request_module();
                    return;
                }
                if (!request->error.empty() || !request->module) {
                    LOG_ERROR(Render_Vulkan, "Async {} shader {:#x} permutation {} failed: {}",
                              request->stage, request->hash, request->permutation_index,
                              request->error);
                    destroy_request_module();
                    return;
                }

                request->specialization.info = &program.info;
                RegisterShaderBinary(std::move(request->spirv), request->hash,
                                     request->permutation_index);
                RegisterShaderMeta(program.info, request->specialization.fetch_shader_data,
                                   request->specialization, permutation_hash,
                                   request->permutation_index);
                module = Program::Module{request->module, std::move(request->specialization)};
                request->module = nullptr;
            });
        });
    };

    auto it_pgm = program_cache.find(params.hash);
    if (it_pgm == program_cache.end()) {
        if (async_compiler && stage != Shader::Stage::Compute) {
            if (async_pending_programs.emplace(params.hash).second) {
                queue_async_shader(true, 0, nullptr);
            }
            graphics_compilation_pending = true;
            return std::nullopt;
        }

        auto [new_it, new_program] = program_cache.try_emplace(params.hash);
        ASSERT(new_program);
        it_pgm = new_it;
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        return Result{&program->info, module, program->modules[0].spec.fetch_shader_data,
                      perm_hash};
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        if (async_compiler && stage != Shader::Stage::Compute) {
            queue_async_shader(false, perm_idx, &spec);
            program->AddPendingPermut(std::move(spec));
            graphics_compilation_pending = true;
            return std::nullopt;
        }

        auto new_info = Shader::Info(stage, l_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        if (!it->module) {
            graphics_compilation_pending = true;
            return std::nullopt;
        }
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    return Result{&program->info, module, program->modules[perm_idx].spec.fetch_shader_data,
                  perm_hash};
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
