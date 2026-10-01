// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "core/performance_telemetry.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/quad_indices.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

namespace {

struct HostImportStateV159 {
    u32 write_faults{};
    bool attempted{};
};

constexpr u32 HostImportWriteFaultThresholdV159 = 1;
constexpr vk::BufferUsageFlags HostImportedBufferFlagsV159 =
    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
std::unordered_map<VAddr, HostImportStateV159> host_import_states_v159;

bool IsKosmicKrisp(const Vulkan::Instance& instance) {
    return instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp;
}

bool IsLbp3KosmicKrisp(const Vulkan::Instance& instance) {
    return Common::ElfInfo::Instance().GameSerial() == "CUSA00063" && IsKosmicKrisp(instance);
}

constexpr std::array Lbp3NgDirectBackingSizes{0x21a8000ULL, 0x0a4000ULL};

bool IsLbp3NgDirectBackingCandidate(const Vulkan::Instance& instance, u64 size) {
    return IsLbp3KosmicKrisp(instance) && std::ranges::contains(Lbp3NgDirectBackingSizes, size);
}

} // namespace

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t QuadIndexBufferSize = 4_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize,
                     Core::PerfTelemetry::GpuWaitResource::UploadStaging},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize,
                    Core::PerfTelemetry::GpuWaitResource::UniformStream},
      quad_index_buffer{instance, scheduler, MemoryUsage::Stream, QuadIndexBufferSize,
                        Core::PerfTelemetry::GpuWaitResource::QuadIndex},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize,
                      Core::PerfTelemetry::GpuWaitResource::Download},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize,
                    Core::PerfTelemetry::GpuWaitResource::DeviceUtility},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    host_import_states_v159.clear();
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), quad_index_buffer.Handle(),
                          "QuadList Index Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);
    bda_pagetable_buffer.Fill(0, BDA_PAGETABLE_SIZE, 0);

    // A direct non-indexed QuadList is exactly two triangles per four vertices. Keep one immutable
    // relative index buffer for every draw instead of asking the driver to tessellate every quad.
    const u32 quad_index_count = GetQuadIndexCount();
    const auto [mapped_indices, index_offset] =
        quad_index_buffer.Map(u64{quad_index_count} * sizeof(u32), alignof(u32));
    ASSERT_MSG(mapped_indices != nullptr && index_offset == 0,
               "Failed to initialize the QuadList index buffer");
    auto* indices = reinterpret_cast<u32*>(mapped_indices);
    for (u32 index = 0, vertex = 0; index < quad_index_count; index += 6, vertex += 4) {
        indices[index + 0] = vertex + 1;
        indices[index + 1] = vertex + 2;
        indices[index + 2] = vertex + 0;
        indices[index + 3] = vertex + 2;
        indices[index + 4] = vertex + 3;
        indices[index + 5] = vertex + 0;
    }
    quad_index_buffer.Commit();

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool download) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    if (download) {
        memory_tracker->InvalidateRegion(
            device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
    } else {
        memory_tracker->InvalidateRegion(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];

        // GPU-modified ranges come as many small scattered islands, so the download
        // is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr buf_start = buffer.CpuAddr();
        const VAddr buf_end = buf_start + buffer.SizeBytes();
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), buf_end);

        if (buffer.IsHostImported()) {
            if (!is_write) {
                DownloadBufferMemory<false>(buffer, window_start, window_end - window_start);
                return;
            }

            // Imported coherent backing is already the guest allocation. A CPU
            // overwrite therefore needs no copy and, crucially, must not mutate the
            // buffer-cache CPU/GPU dirty bits: HostImported buffers are deliberately
            // outside page-fault tracking after promotion. The next GPU bind emits a
            // conservative host/device barrier.
            gpu_modified_ranges.Subtract(device_addr, size);
            buffer.MarkHostWrite();
            return;
        }

        // KosmicKrisp backs imported host allocations with IOSurfaces whose Metal
        // mappings are retained far beyond the VkBuffer lifetime. Even a handful
        // of promotions can consume several gigabytes, so keep that driver on the
        // regular buffer-cache path. Native Vulkan drivers retain host import.
        if (is_write && !IsKosmicKrisp(instance)) {
            auto& state = host_import_states_v159[buffer.CpuAddr()];
            if (!state.attempted && ++state.write_faults >= HostImportWriteFaultThresholdV159) {
                state.attempted = true;
                const bool fully_mapped =
                    memory->IsValidMapping(buffer.CpuAddr(), buffer.SizeBytes());
                if (instance.SupportsExternalMemoryHost() && fully_mapped) {
                    // Make guest backing authoritative once before binding it directly.
                    // DownloadBufferMemory can return without submitting when its exact
                    // dirty-range set is empty. Drain unconditionally before replacing
                    // the old VkBuffer so no queued work can still reference it.
                    DownloadBufferMemory<false>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
                    scheduler.Finish();
                    scheduler.PopPendingOperations();
                    memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                    if (auto retired_backing =
                            buffer.TryImportHostMemory(HostImportedBufferFlagsV159)) {
                        const u64 page_begin = buffer.CpuAddr() >> CACHING_PAGEBITS;
                        const u64 page_count =
                            Common::DivCeil(buffer.SizeBytes(), CACHING_PAGESIZE);
                        std::vector<vk::DeviceAddress> bda_addrs(page_count);
                        for (u64 page = 0; page < page_count; ++page) {
                            bda_addrs[page] =
                                buffer.BufferDeviceAddress() + (page << CACHING_PAGEBITS);
                        }
                        WriteDataBuffer(bda_pagetable_buffer,
                                        page_begin * sizeof(vk::DeviceAddress), bda_addrs.data(),
                                        bda_addrs.size() * sizeof(vk::DeviceAddress));
                        // TryImportHostMemory deliberately transfers ownership of the old
                        // backing here.  WriteDataBuffer only records the BDA page-table
                        // upload, so submit and complete it before retired_backing leaves
                        // scope. Otherwise a shader can still fetch the old BDA after its
                        // VkBuffer/VMA allocation has already been destroyed.
                        scheduler.Finish();
                        scheduler.PopPendingOperations();
                        // A synchronous CPU fault can interrupt descriptor construction:
                        // the rasterizer may still publish a handle captured before the
                        // backing swap after this callback returns. Defer destruction on
                        // the fresh post-Finish tick so the old allocation survives that
                        // next submission, without retaining it for the process lifetime.
                        scheduler.DeferOperation(
                            [backing = std::move(*retired_backing)]() mutable {});
                        return;
                    }
                    // The full synchronous download already serviced this exact write
                    // fault.
                    return;
                }
            }
        }
        DownloadBufferMemory<false>(buffer, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    });
}

void BufferCache::NotifyCpuWrite(VAddr device_addr, u64 size) {
    if (size == 0 || !IsRegionRegistered(device_addr, size)) {
        texture_cache.InvalidateMemory(device_addr, size);
        return;
    }

    const BufferId buffer_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
    if (!IsBufferInvalid(buffer_id)) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsHostImported() && buffer.IsInBounds(device_addr, size)) {
            gpu_modified_ranges.Subtract(device_addr, size);
            buffer.MarkHostWrite();
            texture_cache.InvalidateMemory(device_addr, size);
            return;
        }
    }

    // CPU HLE writes through the physical guest backing, bypassing the protected virtual alias.
    // Reproduce the normal write-fault state transition without downloading bytes that have just
    // been replaced by the CPU implementation.
    InvalidateMemory(device_addr, size, false);
    texture_cache.InvalidateMemory(device_addr, size);
}

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size) {
    const bool direct_target = IsLbp3NgDirectBackingCandidate(instance, buffer.SizeBytes());
    if (buffer.IsHostImported()) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        if (const auto barrier = buffer.GetBarrier(vk::AccessFlagBits2::eHostRead,
                                                   vk::PipelineStageFlagBits2::eHost)) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &barrier.value(),
            });
        }
        scheduler.DeferOperation(
            [this, device_addr, size] { gpu_modified_ranges.Subtract(device_addr, size); });
        if constexpr (!async) {
            if (direct_target) {
                Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectFaultSubmits);
                Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectFaultFinishes);
            }
            scheduler.Finish();
            scheduler.PopPendingOperations();
        }
        return;
    }

    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    if (direct_target) {
        u64 copied_bytes = 0;
        for (const auto& copy : copies) {
            copied_bytes += copy.size;
        }
        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectReadbackBytes,
                                       copied_bytes);
    }
    const VAddr buffer_addr = buffer.CpuAddr();

    // A single GPU-modified cache buffer can be larger than the fixed download
    // ring. Split every range into bounded transfers instead of accepting
    // StreamBuffer::Map's null result and then recording an out-of-bounds copy.
    // Each CPU callback is queued immediately so a ring wrap can drain it before
    // reusing the same mapped bytes.
    for (const auto& copy : copies) {
        u64 source_offset = copy.srcOffset;
        u64 remaining_size = copy.size;
        while (remaining_size != 0) {
            const u64 chunk_size = std::min<u64>(remaining_size, DownloadBufferSize);
            const auto [download, download_offset] = download_buffer.Map(chunk_size, 64);
            if (download == nullptr) {
                for (const auto& pending_copy : copies) {
                    gpu_modified_ranges.Add(buffer_addr + pending_copy.srcOffset,
                                            pending_copy.size);
                }
                LOG_ERROR(Render_Vulkan,
                          "Failed to reserve bounded buffer readback: requested={:#x} "
                          "ring_size={:#x}",
                          chunk_size, DownloadBufferSize);
                return;
            }
            download_buffer.Commit();

            scheduler.EndRendering();
            const auto cmdbuf = scheduler.CommandBuffer();
            const std::array chunk_copy = {vk::BufferCopy{
                .srcOffset = source_offset,
                .dstOffset = download_offset,
                .size = chunk_size,
            }};
            cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), chunk_copy);
            const vk::BufferMemoryBarrier2 host_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                .buffer = download_buffer.Handle(),
                .offset = download_offset,
                .size = chunk_size,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &host_barrier,
            });

            const VAddr copy_device_addr = buffer_addr + source_offset;
            scheduler.DeferOperation([download, copy_device_addr, chunk_size] {
                auto* memory = Core::Memory::Instance();
                memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download, chunk_size);
            });

            source_offset += chunk_size;
            remaining_size -= chunk_size;
        }
    }

    scheduler.DeferOperation([this, device_addr, size] {
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
    });

    if constexpr (!async) {
        if (direct_target) {
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectFaultSubmits);
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectFaultFinishes);
        }
        scheduler.Finish();
        scheduler.PopPendingOperations();
    }
}

void BufferCache::ReadEdgeImagePages(const Image& image) {
    const VAddr image_addr = image.info.guest_address;
    const VAddr image_end = image_addr + image.info.guest_size;
    const VAddr page_start = PageManager::GetPageAddr(image_addr);
    const VAddr page_end = PageManager::GetNextPageAddr(image_end - 1);

    boost::container::small_vector<std::pair<VAddr, VAddr>, 2> ranges;
    const auto collect_range = [&](VAddr start, VAddr end) { ranges.emplace_back(start, end); };
    if (page_start < image_addr) {
        gpu_modified_ranges.ForEachInRange(page_start, image_addr - page_start, collect_range);
    }
    if (image_end < page_end) {
        gpu_modified_ranges.ForEachInRange(image_end, page_end - image_end, collect_range);
    }
    if (ranges.empty()) {
        return;
    }

    Buffer* buffer = ObtainBufferForImage(image_addr, image.info.guest_size).first;
    boost::container::small_vector<vk::BufferCopy, 2> copies;
    u64 total_size_bytes = 0;
    for (const auto [start, end] : ranges) {
        const u64 range_size = end - start;
        ASSERT(buffer->IsInBounds(start, range_size));
        copies.push_back(vk::BufferCopy{
            .srcOffset = buffer->Offset(start),
            .dstOffset = total_size_bytes,
            .size = range_size,
        });
        total_size_bytes = Common::AlignUp(total_size_bytes + range_size, 64ULL);
    }

    const auto [download, download_offset] = download_buffer.Map(total_size_bytes);
    ASSERT(download != nullptr);
    for (auto& copy : copies) {
        copy.dstOffset += download_offset;
    }
    download_buffer.Commit();
    gpu_modified_ranges.Subtract(page_start, page_end - page_start);

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    if (const auto source_barrier = buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                                       vk::PipelineStageFlagBits2::eTransfer)) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &source_barrier.value(),
        });
    }
    const vk::BufferMemoryBarrier2 download_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = download_buffer.Handle(),
        .offset = download_offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &download_barrier,
    });
    cmdbuf.copyBuffer(buffer->Handle(), download_buffer.Handle(), copies);
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_buffer.Handle(),
        .offset = download_offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    scheduler.DeferOperation(
        [copies = std::move(copies), download, download_offset, buffer_addr = buffer->CpuAddr()] {
            auto* memory = Core::Memory::Instance();
            for (const auto& copy : copies) {
                const VAddr device_addr = buffer_addr + copy.srcOffset;
                const u64 data_offset = copy.dstOffset - download_offset;
                memory->TryWriteBacking(std::bit_cast<u8*>(device_addr), download + data_offset,
                                        copy.size);
            }
        });
}

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

std::optional<BufferCache::ExpandedQuadIndices> BufferCache::TryExpandQuadIndices(
    u32 index_offset) {
    const auto& regs = liverpool->regs;
    if (regs.index_buffer_type.swap_mode != AmdGpu::IndexSwapMode::None ||
        (regs.enable_primitive_id & 1) != 0) {
        return std::nullopt;
    }
    const auto type = regs.index_buffer_type.index_type;
    if (type != AmdGpu::IndexType::Index16 && type != AmdGpu::IndexType::Index32) {
        return std::nullopt;
    }
    const u64 output_count = QuadListIndexCount(regs.num_indices);
    if (!output_count || output_count > GetQuadIndexCount()) {
        return std::nullopt;
    }
    const bool index16 = type == AmdGpu::IndexType::Index16;
    const u32 index_size = index16 ? sizeof(u16) : sizeof(u32);
    const VAddr address = regs.index_base_address.Address<VAddr>() + u64{index_offset} * index_size;
    const u64 source_bytes = u64{regs.num_indices / 4} * 4 * index_size;
    // Imported memory deliberately bypasses GPU dirty tracking. It must also be
    // excluded: coherent backing does not imply that a queued GPU write is done.
    if (IsRegionGpuModified(address, source_bytes) || IsHostImportedRange(address, source_bytes)) {
        return std::nullopt;
    }
    const Core::PerfTelemetry::ScopedTimer timer{
        Core::PerfTelemetry::TimeMetric::QuadIndexExpandCpu};
    const auto [mapped, offset] = stream_buffer.Map(output_count * index_size, index_size, false);
    if (!mapped) {
        return std::nullopt;
    }
    memory->CopySparseMemory(address, mapped, source_bytes);
    const bool restart_enabled = (regs.enable_primitive_restart & 1) != 0;
    const u32 count =
        index16
            ? ExpandQuadIndicesInPlace(
                  std::span{reinterpret_cast<u16*>(mapped), size_t(output_count)}, regs.num_indices,
                  restart_enabled ? std::optional<u16>(regs.primitive_restart_index) : std::nullopt)
            : ExpandQuadIndicesInPlace(
                  std::span{reinterpret_cast<u32*>(mapped), size_t(output_count)}, regs.num_indices,
                  restart_enabled ? std::optional<u32>(regs.primitive_restart_index)
                                  : std::nullopt);
    stream_buffer.Commit();
    if (count == 0) {
        return std::nullopt;
    }
    ASSERT(count == output_count);
    Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::ExpandedQuadIndexBytes,
                                   output_count * index_size);
    return ExpandedQuadIndices{stream_buffer.Handle(), offset, count,
                               index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32};
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(vk_buffer->Handle(), offset, index_type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes) && !IsHostImportedRange(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

bool BufferCache::WriteGuestFence(VAddr address, u64 value, u32 num_bytes) {
    if (!IsLbp3KosmicKrisp(instance) || !instance.SupportsExternalMemoryHost() ||
        (num_bytes != sizeof(u32) && num_bytes != sizeof(u64)) || address % alignof(u32) != 0 ||
        !memory->IsValidMapping(address, num_bytes)) {
        return false;
    }

    BufferId buffer_id = page_table[address >> CACHING_PAGEBITS].buffer_id;
    if (IsBufferInvalid(buffer_id) || !slot_buffers[buffer_id].IsInBounds(address, num_bytes) ||
        !slot_buffers[buffer_id].IsHostImported()) {
        // RELEASE_MEM labels are tiny and normally have no cache allocation at
        // all. If the page already belongs to a device-local buffer, replace it
        // with a joined imported incarnation before recording the first write.
        buffer_id = CreateBuffer(address, num_bytes, true);
    }

    Buffer& buffer = slot_buffers[buffer_id];
    if (!buffer.IsHostImported() || !buffer.IsInBounds(address, num_bytes)) {
        return false;
    }

    SynchronizeBuffer(buffer, address, num_bytes, true, false);
    const u32 offset = buffer.Offset(address);
    buffer.Fill(offset, sizeof(u32), static_cast<u32>(value));
    if (num_bytes == sizeof(u64)) {
        buffer.Fill(offset + sizeof(u32), sizeof(u32), static_cast<u32>(value >> 32));
    }
    return true;
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes) && !IsHostImportedRange(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !IsHostImportedRange(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers are still in guest memory. Do not use a raw host memcpy
            // here: DMA ranges can straddle sparse/unbacked VMAs even when their
            // first byte is valid.
            const auto valid_prefix = [this](VAddr address, u32 requested_size) {
                if (memory->IsValidMapping(address, requested_size)) {
                    return requested_size;
                }

                // IsValidMapping is monotonic for a prefix beginning at a fixed
                // address.
                u32 valid_size = 0;
                u32 invalid_size = requested_size;
                while (valid_size < invalid_size) {
                    const u32 candidate = valid_size + (invalid_size - valid_size + 1) / 2;
                    if (memory->IsValidMapping(address, candidate)) {
                        valid_size = candidate;
                    } else {
                        invalid_size = candidate - 1;
                    }
                }
                return valid_size;
            };

            const u32 copy_size =
                std::min(valid_prefix(src, num_bytes), valid_prefix(dst, num_bytes));
            if (copy_size != num_bytes) {
                LOG_WARNING(Render_Vulkan,
                            "Clamping sparse CPU buffer copy src={:#x} dst={:#x} "
                            "requested={:#x} "
                            "to mapped prefix {:#x}",
                            src, dst, num_bytes, copy_size);
            }
            if (copy_size == 0) {
                return;
            }

            // TryWriteBacking bypasses the protected virtual alias, so reproduce the
            // invalidation which a raw guest write fault would normally trigger
            // before changing the backing.
            InvalidateMemory(dst, copy_size);
            texture_cache.InvalidateMemory(dst, copy_size);

            constexpr u32 CopyChunkSize = 64_KB;
            std::vector<u8> copy_data(std::min(copy_size, CopyChunkSize));
            const bool copy_backward = dst > src && dst - src < copy_size;
            for (u32 bytes_left = copy_size; bytes_left != 0;) {
                const u32 chunk_size =
                    std::min<u32>(bytes_left, static_cast<u32>(copy_data.size()));
                const u32 offset = copy_backward ? bytes_left - chunk_size : copy_size - bytes_left;
                memory->CopySparseMemory(src + offset, copy_data.data(), chunk_size);
                if (!memory->TryWriteBacking(std::bit_cast<void*>(dst + offset), copy_data.data(),
                                             chunk_size)) {
                    LOG_WARNING(Render_Vulkan,
                                "Sparse CPU buffer copy destination has no physical backing: "
                                "src={:#x} dst={:#x} offset={:#x} size={:#x}",
                                src, dst, offset, chunk_size);
                    break;
                }
                bytes_left -= chunk_size;
            }
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        texture_cache.InvalidateTexelBufferSync(dst, num_bytes);
        if (!buffer.IsHostImported()) {
            gpu_modified_ranges.Add(dst, num_bytes);
        }
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  bool invalidate_texel_sync) {
    // For read-only buffers use device local stream buffer to reduce renderpass
    // breaks.
    if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(device_addr, size) &&
        !IsHostImportedRange(device_addr, size)) {
        const u64 offset = stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
        return {&stream_buffer, offset};
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        if (invalidate_texel_sync) {
            texture_cache.InvalidateTexelBufferSync(device_addr, size);
        }
        if (!buffer.IsHostImported()) {
            gpu_modified_ranges.Add(device_addr, size);
        }
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing
    // GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::PreserveImage(Image& image) {
    const auto [buffer, offset] =
        ObtainBuffer(image.info.guest_address, image.info.guest_size, true, false, {}, false);
    ASSERT(offset == buffer->Offset(image.info.guest_address));
    return SynchronizeBufferFromImage(*buffer, image);
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsHostImportedRange(VAddr device_addr, u64 size) {
    if (device_addr == 0 || size == 0) {
        return false;
    }
    const BufferId buffer_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
    if (!buffer_id || slot_buffers[buffer_id].is_deleted) {
        return false;
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    return buffer.IsHostImported() && buffer.IsInBounds(device_addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume
            // it's being used as a stream buffer. Increase the size to skip
            // constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    if (new_buffer.IsHostImported() && overlap.IsHostImported()) {
        // Both Vulkan objects directly view the same guest bytes. Copying between
        // them would be both redundant and an overlapping alias operation.
        DeleteBuffer(overlap_id);
        return;
    }
    if (new_buffer.IsHostImported()) {
        // CPU-authoritative bytes are already present in the imported destination.
        // Migrate only ranges whose newest representation is still in the old GPU
        // buffer, otherwise a full copy would overwrite newer guest writes.
        gpu_modified_ranges.ForEachInRange(overlap.CpuAddr(), overlap.SizeBytes(),
                                           [&](VAddr begin, VAddr end) {
                                               copies.push_back(vk::BufferCopy{
                                                   .srcOffset = begin - overlap.CpuAddr(),
                                                   .dstOffset = begin - new_buffer.CpuAddr(),
                                                   .size = end - begin,
                                               });
                                           });
    } else {
        copies.push_back(vk::BufferCopy{
            .srcOffset = 0,
            .dstOffset = dst_base_offset,
            .size = overlap.SizeBytes(),
        });
    }
    if (copies.empty()) {
        DeleteBuffer(overlap_id);
        return;
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copies);
    if (new_buffer.IsHostImported()) {
        for (const auto& copy : copies) {
            TrackHostImportedWrite(new_buffer, new_buffer.CpuAddr() + copy.dstOffset, copy.size);
        }
    }

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size, bool force_direct_import) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    const bool direct_candidate = IsLbp3NgDirectBackingCandidate(instance, size) ||
                                  (force_direct_import && IsLbp3KosmicKrisp(instance));
    if (direct_candidate) {
        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectImportAttempts);
    }
    if (direct_candidate && instance.SupportsExternalMemoryHost() &&
        memory->IsValidMapping(overlap.begin, size)) {
        // This device-local allocation has not been referenced by any recorded
        // command yet, so the unused backing returned by TryImportHostMemory can
        // be destroyed immediately. The imported object is the buffer's first and
        // only published BDA/descriptor backing.
        // Existing overlap buffers can leave these pages read-protected or
        // completely inaccessible to the CPU. Metal's newBufferWithBytesNoCopy
        // rejects PROT_NONE memory, so relax only the OS protection while it
        // validates/retains the mapping. Watcher counts and dirty ranges remain
        // intact and are used by JoinOverlap below.
        memory_tracker->RelaxProtectionForHostImport(overlap.begin, size);
        auto unused_backing = new_buffer.TryImportHostMemory(HostImportedBufferFlagsV159);
        memory_tracker->RefreshProtectionAfterHostImport(overlap.begin, size);
        if (unused_backing) {
            LOG_INFO(Render_Vulkan, "LBP3 NG direct backing enabled: base={:#x} size={:#x}",
                     overlap.begin, size);
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectImportSuccesses);
        } else {
            LOG_WARNING(Render_Vulkan,
                        "LBP3 NG direct backing import failed: base={:#x} size={:#x}",
                        overlap.begin, size);
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectImportFailures);
        }
    } else if (direct_candidate) {
        LOG_WARNING(Render_Vulkan,
                    "LBP3 NG direct backing unavailable: base={:#x} size={:#x} "
                    "external_host={} fully_mapped={}",
                    overlap.begin, size, instance.SupportsExternalMemoryHost(),
                    memory->IsValidMapping(overlap.begin, size));
        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectImportFailures);
    }
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    if (new_buffer.IsHostImported()) {
        // Host-imported buffers deliberately live outside the page-fault dirty
        // protocol. Remove stale state inherited from previous buffer
        // incarnations before exposing the final BDA.
        memory_tracker->InvalidateRegion(overlap.begin, size);
        gpu_modified_ranges.Subtract(overlap.begin, size);
    }
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    if (buffer.IsHostImported()) {
        if (IsLbp3NgDirectBackingCandidate(instance, buffer.SizeBytes())) {
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectBufferBinds);
        }
        // The imported allocation and guest allocation are the same coherent
        // bytes. Do not consume CPU dirty ranges or create GPU dirty page tracking;
        // both operations would re-arm synchronous page faults and can unbalance
        // read watchers. Since host writes are intentionally untracked, emit this
        // availability/visibility barrier on every bind.
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        vk::AccessFlags2 destination_access = vk::AccessFlagBits2::eMemoryRead;
        if (is_written) {
            destination_access |= vk::AccessFlagBits2::eMemoryWrite;
        }
        const auto barrier =
            buffer.GetBarrier(destination_access, vk::PipelineStageFlagBits2::eAllCommands,
                              buffer.Offset(device_addr));
        ASSERT(barrier.has_value());
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
        if (is_written) {
            TrackHostImportedWrite(buffer, device_addr, size);
        }
        TouchBuffer(buffer);
        if (is_texel_buffer) {
            return SynchronizeBufferFromImage(buffer, device_addr, size);
        }
        return false;
    }

    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    if (src_buffer) {
        if (IsLbp3NgDirectBackingCandidate(instance, buffer.SizeBytes())) {
            Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectUploadBytes,
                                           total_size_bytes);
        }
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        TouchBuffer(buffer);
    }
    if (is_texel_buffer) {
        // Formatted storage buffers may perform read-modify-write operations.
        // Preserve an overlapping GPU-modified image before exposing its raw tiled
        // backing to the shader.
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

void BufferCache::TrackHostImportedWrite(const Buffer& buffer, VAddr device_addr, u64 size) {
    ASSERT(buffer.IsHostImported());
    ASSERT(buffer.IsInBounds(device_addr, size));
    const u64 begin = buffer.Offset(device_addr);
    const u64 end = begin + size;
    for (auto& range : pending_host_imported_writes) {
        if (range.buffer != buffer.Handle()) {
            continue;
        }
        range.begin = std::min(range.begin, begin);
        range.end = std::max(range.end, end);
        return;
    }
    pending_host_imported_writes.push_back({buffer.Handle(), begin, end});
}

bool BufferCache::CommitHostImportedWritesForCpu() {
    if (pending_host_imported_writes.empty()) {
        return false;
    }

    scheduler.EndRendering();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 4> barriers;
    barriers.reserve(pending_host_imported_writes.size());
    for (const auto& range : pending_host_imported_writes) {
        barriers.push_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eHostWrite,
            .buffer = range.buffer,
            .offset = range.begin,
            .size = range.end - range.begin,
        });
    }
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
    });
    Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::DirectVisibilityBarriers,
                                   barriers.size());
    pending_host_imported_writes.clear();
    return true;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const Core::PerfTelemetry::ScopedGpuWaitContext upload_context{
        Core::PerfTelemetry::GpuWaitContext::BufferUpload};
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    const u32 required_size = std::min(size, image.info.guest_size);
    if (image.IsTexelBufferSynced(buffer.Handle(), buf_offset, required_size)) {
        TouchBuffer(buffer);
        return true;
    }
    return SynchronizeBufferFromImage(buffer, image);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, Image& image) {
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    u32 sync_size = 0;
    for (u32 mip = 0; mip < image.info.guest_resources.levels; mip++) {
        const auto& mip_info = image.info.guest_mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.guest_resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
        sync_size = std::max(sync_size, mip_info.offset + mip_info.size);
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    if (buffer.IsHostImported()) {
        TrackHostImportedWrite(buffer, image.info.guest_address, sync_size);
    }
    image.MarkTexelBufferSynced(buffer.Handle(), buf_offset, sync_size);
    TouchBuffer(buffer);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
        memory_tracker->MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        DeleteBuffer(buffer_id);
    };
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    host_import_states_v159.erase(buffer.CpuAddr());
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

} // namespace VideoCore
