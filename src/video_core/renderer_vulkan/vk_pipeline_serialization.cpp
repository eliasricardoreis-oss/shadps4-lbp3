// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/scope_exit.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <chrono>
#include <type_traits>

#ifdef __APPLE__
#include "sdl_window.h"
extern Frontend::WindowSDL* g_window;
#endif

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 2u;
static constexpr u32 ShaderMetaVersion = 2u;
static constexpr u32 PipelineKeyVersion = 3u;
} // namespace Serialization

namespace Vulkan {

namespace {

constexpr u64 FnvOffset = 14695981039346656037ULL;
constexpr u64 FnvPrime = 1099511628211ULL;

template <typename T>
void HashProfileValue(u64& hash, T value) {
    if constexpr (std::is_same_v<T, bool>) {
        hash ^= static_cast<u8>(value ? 1 : 0);
        hash *= FnvPrime;
    } else {
        using Unsigned = std::make_unsigned_t<T>;
        auto bits = static_cast<Unsigned>(value);
        for (size_t index = 0; index < sizeof(bits); ++index) {
            hash ^= static_cast<u8>(bits & 0xff);
            hash *= FnvPrime;
            bits >>= 8;
        }
    }
}

u64 StableProfileHash(const Shader::Profile& profile) {
    u64 hash = FnvOffset;
#define HASH_PROFILE_FIELD(field) HashProfileValue(hash, profile.field)
    HASH_PROFILE_FIELD(max_ubo_size);
    HASH_PROFILE_FIELD(max_viewport_width);
    HASH_PROFILE_FIELD(max_viewport_height);
    HASH_PROFILE_FIELD(max_shared_memory_size);
    HASH_PROFILE_FIELD(supported_spirv);
    HASH_PROFILE_FIELD(subgroup_size);
    HASH_PROFILE_FIELD(support_int8);
    HASH_PROFILE_FIELD(support_int16);
    HASH_PROFILE_FIELD(support_int64);
    HASH_PROFILE_FIELD(support_float16);
    HASH_PROFILE_FIELD(support_float64);
    HASH_PROFILE_FIELD(supports_denorm_behavior_independence);
    HASH_PROFILE_FIELD(supports_rounding_mode_independence);
    HASH_PROFILE_FIELD(support_fp16_denorm_preserve);
    HASH_PROFILE_FIELD(support_fp16_denorm_flush);
    HASH_PROFILE_FIELD(support_fp16_round_to_zero);
    HASH_PROFILE_FIELD(support_fp32_denorm_preserve);
    HASH_PROFILE_FIELD(support_fp32_denorm_flush);
    HASH_PROFILE_FIELD(support_fp32_round_to_zero);
    HASH_PROFILE_FIELD(support_fp64_denorm_preserve);
    HASH_PROFILE_FIELD(support_fp64_denorm_flush);
    HASH_PROFILE_FIELD(support_fp64_round_to_zero);
    HASH_PROFILE_FIELD(support_fp16_signed_zero_inf_nan_preserve);
    HASH_PROFILE_FIELD(support_fp32_signed_zero_inf_nan_preserve);
    HASH_PROFILE_FIELD(support_fp64_signed_zero_inf_nan_preserve);
    HASH_PROFILE_FIELD(supports_image_load_store_lod);
    HASH_PROFILE_FIELD(supports_native_cube_calc);
    HASH_PROFILE_FIELD(supports_trinary_minmax);
    HASH_PROFILE_FIELD(supports_buffer_fp32_atomic_min_max);
    HASH_PROFILE_FIELD(supports_image_fp32_atomic_min_max);
    HASH_PROFILE_FIELD(supports_buffer_int64_atomics);
    HASH_PROFILE_FIELD(supports_shared_int64_atomics);
    HASH_PROFILE_FIELD(supports_workgroup_explicit_memory_layout);
    HASH_PROFILE_FIELD(supports_amd_shader_explicit_vertex_parameter);
    HASH_PROFILE_FIELD(supports_fragment_shader_barycentric);
    HASH_PROFILE_FIELD(has_broken_spirv_clamp);
    HASH_PROFILE_FIELD(lower_left_origin_mode);
    HASH_PROFILE_FIELD(needs_manual_interpolation);
    HASH_PROFILE_FIELD(needs_lds_barriers);
    HASH_PROFILE_FIELD(needs_buffer_offsets);
    HASH_PROFILE_FIELD(needs_unorm_fixup);
    HASH_PROFILE_FIELD(needs_clip_distance_emulation);
    HASH_PROFILE_FIELD(supports_shader_stencil_export);
    HASH_PROFILE_FIELD(supports_shader_output_layer);
#undef HASH_PROFILE_FIELD
    return hash;
}

bool IsCompatibleProfileFile(const std::filesystem::path& path, const Shader::Profile& profile) {
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
    if (!file.IsOpen() || file.GetSize() != sizeof(Shader::Profile)) {
        return false;
    }

    Shader::Profile cached_profile{};
    return file.Read(cached_profile) == 1 && cached_profile == profile;
}

u32 MergeCompatibleProfileDirectories(const std::filesystem::path& root,
                                      std::string_view profile_prefix,
                                      std::string_view canonical_namespace,
                                      const Shader::Profile& profile) {
    namespace fs = std::filesystem;

    std::error_code error;
    if (!fs::is_directory(root, error)) {
        return 0;
    }

    const auto canonical_path = root / canonical_namespace;
    fs::create_directories(canonical_path, error);
    error.clear();

    u32 imported{};
    for (fs::directory_iterator directory{root, error}, end; directory != end;
         directory.increment(error)) {
        if (error) {
            error.clear();
            continue;
        }
        if (!directory->is_directory(error)) {
            error.clear();
            continue;
        }

        const auto source_name = directory->path().filename().string();
        if (source_name == canonical_namespace || !source_name.starts_with(profile_prefix) ||
            !IsCompatibleProfileFile(directory->path() / "profile.bin", profile)) {
            continue;
        }

        for (fs::directory_iterator file{directory->path(), error}, file_end; file != file_end;
             file.increment(error)) {
            if (error) {
                error.clear();
                continue;
            }
            if (!file->is_regular_file(error) || file->path().extension() == ".tmp") {
                error.clear();
                continue;
            }

            const auto destination = canonical_path / file->path().filename();
            if (fs::copy_file(file->path(), destination, fs::copy_options::skip_existing, error)) {
                ++imported;
            }
            error.clear();
        }
    }
    return imported;
}

void MigrateCompatibleLbp3Profiles(std::string_view profile_prefix,
                                   std::string_view canonical_namespace,
                                   const Shader::Profile& profile) {
    const auto title_root = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "CUSA00063";
    u32 imported{};
    for (const auto* side : {"runtime", "seed"}) {
        imported += MergeCompatibleProfileDirectories(title_root / side / "spirv" / "v1",
                                                      profile_prefix, canonical_namespace, profile);
    }
    if (imported != 0) {
        LOG_INFO(Render, "Merged {} compatible LBP3 shader-cache files into stable profile {}",
                 imported, canonical_namespace);
    }
}

} // namespace

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, program->info, fetch_shader, spec, perm_idx)) {
        return false;
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx),
                                       spv);
    if (spv.empty()) {
        return false;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // If the permutation is already preloaded, make sure it has the same permutation index
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render,
                            "Skipping inconsistent cached permutation {} already inserted at {} "
                            "({}_{:x})",
                            perm_idx, idx, program->info.stage, program->info.pgm_hash);
                return false;
            }
            module = it->module;
        } else {
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;

    return true;
}

void PipelineCache::WarmUp() {
    const auto& game_info = Common::ElfInfo::Instance();
    const bool force_lbp3_cache = game_info.GameSerial() == "CUSA00063";
    if (!EmulatorSettings.IsPipelineCacheEnabled() && !force_lbp3_cache) {
        return;
    }

    if (force_lbp3_cache) {
        // Shader::Profile captures the host capabilities and driver workarounds that affect emitted
        // SPIR-V. Keep exact profiles in separate buckets so an M4/M5/Kosmic corpus never poisons a
        // Windows/NVIDIA run (or vice versa), while still using the same title-root layout.
        const auto profile_prefix =
            fmt::format("app-{}-schema-{}-{}-{}-profile-", game_info.AppVer(),
                        Serialization::ShaderBinaryVersion, Serialization::ShaderMetaVersion,
                        Serialization::PipelineKeyVersion);
        const auto profile_namespace =
            fmt::format("{}{:016x}", profile_prefix, StableProfileHash(profile));
        MigrateCompatibleLbp3Profiles(profile_prefix, profile_namespace, profile);
        Storage::DataBase::Instance().ConfigureLbp3Profile(profile_namespace);
        LOG_INFO(Render, "LBP3 persistent shader cache forced on");
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};
    const auto preload_total =
        Storage::DataBase::Instance().CountBlobs(Storage::BlobType::PipelineKey);
    const auto preload_start = std::chrono::steady_clock::now();
#ifdef __APPLE__
    if (g_window && preload_total) {
        Frontend::ShowGraphicsPreparation(g_window->GetSDLWindow(), 0, preload_total);
    }
    SCOPE_EXIT {
        if (g_window && preload_total) {
            Frontend::HideGraphicsPreparation(g_window->GetSDLWindow());
        }
    };
    auto last_progress = preload_start;
#endif
    LOG_INFO(Render, "Preparing {} cached graphics/compute pipelines", preload_total);

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
#ifdef __APPLE__
            const auto now = std::chrono::steady_clock::now();
            if (g_window && now - last_progress >= std::chrono::milliseconds{250}) {
                Frontend::ShowGraphicsPreparation(g_window->GetSDLWindow(), num_total_pipelines,
                                                  preload_total);
                last_progress = now;
            }
#endif
            ++num_total_pipelines;

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });

    LOG_INFO(
        Render, "Preloaded {} pipelines in {:.1f} seconds", num_pipelines,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - preload_start).count());
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    WaitForAsyncCompiler();
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    auto persistent_runtime_info = runtime_info;
    if (persistent_runtime_info.stage == Stage::Geometry) {
        persistent_runtime_info.gs_info.vs_copy = {};
    }
    spec.Write(persistent_runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);
    if (runtime_info.stage == Stage::Geometry) {
        runtime_info.gs_info.vs_copy = {};
    }

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
