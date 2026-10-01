// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/performance_telemetry.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

namespace {

bool IsDimensionalAlias(const ImageInfo& lhs, const ImageInfo& rhs) {
    const bool lhs_is_volume = lhs.type == AmdGpu::ImageType::Color3D;
    const bool rhs_is_volume = rhs.type == AmdGpu::ImageType::Color3D;
    if (lhs.guest_address != rhs.guest_address || lhs_is_volume == rhs_is_volume ||
        lhs.tile_mode != rhs.tile_mode || lhs.array_mode != rhs.array_mode ||
        lhs.pitch != rhs.pitch || lhs.BlockDim() != rhs.BlockDim() ||
        lhs.size.width != rhs.size.width || lhs.size.height != rhs.size.height ||
        lhs.num_bits * lhs.num_samples != rhs.num_bits * rhs.num_samples ||
        !lhs.IsCompatible(rhs)) {
        return false;
    }

    if (lhs.guest_size == rhs.guest_size) {
        return true;
    }

    // A guest can expose only the active prefix of a volume through a 2D-array descriptor. The
    // two descriptors still alias when their per-slice layout is identical; their host images
    // simply have a different number of slices. Restrict unequal ranges to the single-mip case so
    // a partial mip chain cannot be mistaken for a dimensional reinterpretation.
    const auto& volume = lhs_is_volume ? lhs : rhs;
    const auto& array = lhs_is_volume ? rhs : lhs;
    if (volume.guest_resources.levels != 1 || array.guest_resources.levels != 1) {
        return false;
    }

    const u32 volume_slices = std::max(volume.size.depth, 1u);
    const u32 array_slices = std::max(array.guest_resources.layers, 1u);
    const u32 volume_bytes = volume.guest_mips_layout[0].size;
    const u32 array_bytes = array.guest_mips_layout[0].size;
    return volume_bytes == volume.guest_size && array_bytes == array.guest_size &&
           volume_bytes % volume_slices == 0 && array_bytes % array_slices == 0 &&
           volume_bytes / volume_slices == array_bytes / array_slices;
}

bool CanUseVolumeMaster(const ImageInfo& master, const ImageInfo& view) {
    if (master.type != AmdGpu::ImageType::Color3D || view.type == AmdGpu::ImageType::Color3D ||
        !IsDimensionalAlias(master, view)) {
        return false;
    }

    // A 3D Vulkan image has one array layer, while compatible 2D/2D-array views address its
    // depth slices through baseArrayLayer/layerCount. Compare the view's layer requirement with
    // volume depth instead of using SubresourceExtent::Contains on unrelated axes.
    const u32 required_slices = std::max(view.resources.layers, view.guest_resources.layers);
    return master.resources.levels >= view.resources.levels && master.size.depth >= required_slices;
}

} // namespace

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync, bool gc_retirement) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified) ||
        True(image.flags & ImageFlagBits::DimensionalAliasStale)) {
        return;
    }

    const VAddr image_addr = image.info.guest_address;
    const u32 image_size = image.info.guest_size;
    const u64 image_uid = image.image_uid;
    const Core::PerfTelemetry::ScopedGpuWaitContext download_context{
        Core::PerfTelemetry::GpuWaitContext::ImageDownload, image_addr, image_size};
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.guest_resources.levels; ++mip) {
        const auto& mip_info = image.info.guest_mips_layout[mip];
        ASSERT(static_cast<u64>(mip_info.offset) + mip_info.size <= image_size);
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
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
    }
    if (buffer_copies.empty()) {
        return;
    }

    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    // Synchronous readbacks can safely borrow the shared ring because Finish consumes the bytes
    // before this function returns. An asynchronous GC readback must own its mapped allocation
    // until the deferred CPU callback runs; sharing it with buffer-cache readbacks risks reusing
    // the range before the texture snapshot has been consumed.
    const auto stream_mapping =
        sync ? download_buffer.Map(image_size) : std::pair<u8*, u64>{nullptr, 0};
    std::unique_ptr<Buffer> temporary_buffer;
    Buffer* output_buffer = &download_buffer;
    u8* download = stream_mapping.first;
    u64 output_offset = stream_mapping.second;
    if (download) {
        download_buffer.Commit();
    } else {
        temporary_buffer = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                                    AllFlags, image_size);
        output_buffer = temporary_buffer.get();
        download = output_buffer->mapped_data.data();
        output_offset = 0;
        ASSERT(download != nullptr);
    }

    tile_manager.TileImage(image, buffer_copies, output_buffer->Handle(),
                           static_cast<u32>(output_offset), image_size);
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = output_buffer->Handle(),
        .offset = output_offset,
        .size = image_size,
    };
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    auto retirement_ready = gc_retirement ? std::make_shared<bool>(false) : nullptr;
    auto write_back = [this, image_id, image_uid, image_addr, image_size, download, sync,
                       gc_retirement, retirement_ready,
                       temporary_buffer = std::move(temporary_buffer)]() mutable {
        bool wrote_guest = false;
        bool guest_is_current = false;
        if (gc_retirement) {
            if (slot_images.is_allocated(image_id)) {
                Image& current_image = slot_images[image_id];
                if (current_image.image_uid == image_uid &&
                    True(current_image.flags & ImageFlagBits::GcPending)) {
                    if (True(current_image.flags & ImageFlagBits::GpuDirty)) {
                        // A newer copy now lives in the buffer cache. The queued image snapshot
                        // is stale, so keep this image alive and retry after it is synchronized.
                        guest_is_current = false;
                    } else if (True(current_image.flags &
                                    (ImageFlagBits::MaybeCpuDirty | ImageFlagBits::CpuDirty))) {
                        // A CPU write won the race; its guest bytes are already authoritative.
                        guest_is_current = true;
                    } else {
                        wrote_guest = Core::Memory::Instance()->TryWriteBacking(
                            std::bit_cast<u8*>(image_addr), download, image_size);
                        guest_is_current = wrote_guest;
                    }
                }
            }
            *retirement_ready = guest_is_current;
        } else {
            wrote_guest = Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image_addr),
                                                                    download, image_size);
        }
        if (!sync && wrote_guest) {
            buffer_cache.InvalidateMemory(image_addr, image_size, false);
        }
        if (gc_retirement) {
            ASSERT(pending_gc_download_bytes >= image_size);
            pending_gc_download_bytes -= image_size;
        }
        temporary_buffer.reset();
    };

    if (sync) {
        scheduler.Finish();
        write_back();
    } else {
        scheduler.DeferOperation(std::move(write_back));
        if (gc_retirement) {
            scheduler.DeferOperation([this, image_id, image_uid, retirement_ready] {
                if (!slot_images.is_allocated(image_id)) {
                    return;
                }
                Image& current_image = slot_images[image_id];
                if (current_image.image_uid != image_uid ||
                    False(current_image.flags & ImageFlagBits::GcPending)) {
                    return;
                }
                current_image.flags &= ~ImageFlagBits::GcPending;
                if (*retirement_ready) {
                    FreeImage(image_id);
                } else {
                    // The guest mapping changed before the callback. Keep the valid GPU image and
                    // age it again instead of deleting the only authoritative copy.
                    TouchImage(current_image);
                }
            });
        }
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags &= ~ImageFlagBits::DimensionalAliasStale;
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    Core::PerfTelemetry::ScopedFaultLock lock{
        mutex, Core::PerfTelemetry::Counter::TextureFaultLockWaits,
        Core::PerfTelemetry::TimeMetric::TextureFaultLockWait};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        // A guest write makes the shared backing memory authoritative for every dimensional view.
        image.flags &= ~ImageFlagBits::DimensionalAliasStale;
        image.InvalidateTexelBufferSync();
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    if (max_size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        image.InvalidateTexelBufferSync();
        // Page-table candidates can merely share an edge page. Only a real byte-range overlap
        // is affected by the GPU write. This also covers images and subresources which begin
        // inside a larger formatted-buffer or DMA destination.
        if (!image.Overlaps(address, max_size)) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        // The buffer cache now owns the newest common guest representation.
        image.flags &= ~ImageFlagBits::DimensionalAliasStale;
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::InvalidateTexelBufferSync(VAddr address, size_t size) {
    if (size == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    ForEachImageInRegion(address, size,
                         [](ImageId, Image& image) { image.InvalidateTexelBufferSync(); });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache is missing either mip levels or slices we need to expand it.
    bool recreate = !cache_image.info.resources.Contains(requested_info.resources);

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        const auto expanded_resources =
            requested_info.resources.ExpandedWith(cache_image.info.resources);
        if (expanded_resources != requested_info.resources) {
            // Grow only the Vulkan allocation. The requested descriptor remains the sole proof of
            // readable guest layout/size; the cross-product can exceed the actual allocation.
            new_info.ExpandResources(expanded_resources);
        }
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool requested_is_volume = image_info.type == AmdGpu::ImageType::Color3D;
        const bool cached_is_volume = cache_image.info.type == AmdGpu::ImageType::Color3D;
        const bool is_dimensional_alias = IsDimensionalAlias(image_info, cache_image.info);

        // A guest allocation can be sampled both as a 3D volume and as a 2D array. Vulkan does
        // not allow a 2D-array view created from a 3D image to be used as a sampled descriptor,
        // so both host image types are required. FindImage creates the missing interpretation and
        // tracks which resident peer is stale after a GPU write. Never route a compatible
        // dimensional alias through destructive image expansion.
        if (is_dimensional_alias) {
            if (merged_image_id || True(cache_image.flags & ImageFlagBits::DimensionalAliasStale)) {
                return {merged_image_id, -1, -1};
            }
            return {CreateDimensionalAlias(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (requested_is_volume || cached_is_volume)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool is_compatible =
            IsVulkanFormatCompatible(cache_image.info.pixel_format, image_info.pixel_format);

        // Expand before the guest-size shortcut below: subresource containment is a
        // component-wise relationship, not a comparison of the total byte sizes. If one request
        // adds mip levels while the other adds array layers, preserve both axes in the new backing
        // image instead of selecting one descriptor lexicographically.
        if (is_compatible && image_info.type == cache_image.info.type &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            !cache_image.info.resources.Contains(image_info.resources)) {
            auto expanded_info = image_info;
            const auto expanded_resources =
                image_info.resources.ExpandedWith(cache_image.info.resources);
            if (expanded_resources != image_info.resources) {
                // Preserve the requested descriptor's guest range while expanding host capacity.
                expanded_info.ExpandResources(expanded_resources);
            }
            if (!image_info.resources.Contains(cache_image.info.resources)) {
                LOG_INFO(Render_Vulkan,
                         "Expanding incomparable image resources at {:#x}: cached M:{} L:{}, "
                         "requested M:{} L:{}, union M:{} L:{}",
                         image_info.guest_address, cache_image.info.resources.levels,
                         cache_image.info.resources.layers, image_info.resources.levels,
                         image_info.resources.layers, expanded_info.resources.levels,
                         expanded_info.resources.layers);
            }
            return {ExpandImage(expanded_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool result_is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            const bool result_contains_request =
                result_image.info.resources.Contains(image_info.resources);
            return {result_is_compatible && result_contains_request ? result_id : ImageId{}, -1,
                    -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources contain new: {} (old: M:{} L:{}, new: M:{} L:{})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  cache_image.info.resources.Contains(image_info.resources),
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  image_info.resources.levels, image_info.resources.layers,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];
    const bool source_is_dimensional_alias_stale =
        True(src_image.flags & ImageFlagBits::DimensionalAliasStale);

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    if (source_is_dimensional_alias_stale) {
        new_image.flags |= ImageFlagBits::DimensionalAliasStale;
    }
    return new_image_id;
}

ImageId TextureCache::CreateDimensionalAlias(const ImageInfo& info, ImageId source_id) {
    const bool promote_to_volume_master =
        instance.Is2dViewOf3dSupported() && info.type == AmdGpu::ImageType::Color3D &&
        slot_images[source_id].info.type != AmdGpu::ImageType::Color3D;
    const auto alias_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(alias_id);

    auto& source = slot_images[source_id];
    auto& alias = slot_images[alias_id];
    // Populate destination-only slices from guest memory before the authoritative peer overwrites
    // the common prefix. This matters for partial 2D-array views of a larger 3D volume.
    RefreshImage(alias);
    RefreshImage(source);
    alias.CopyImage(source);
    alias.flags &=
        ~(ImageFlagBits::Dirty | ImageFlagBits::GpuModified | ImageFlagBits::DimensionalAliasStale);
    alias.flags |= source.flags & ImageFlagBits::GpuModified;
    alias.mip_hashes = source.mip_hashes;

    if (promote_to_volume_master) {
        // On hosts with VK_EXT_image_2d_view_of_3d, keep one authoritative 3D allocation and use
        // its compatible 2D slice views for guest render targets/storage descriptors. This makes
        // feedback-loop detection and ordering operate on the actual shared resource instead of
        // copying temporal lighting data between two independent host images every draw.
        alias.usage = source.usage;
        if (source.binding.is_bound || source.binding.is_target) {
            source.binding.needs_rebind = 1u;
        }

        FreeImage(source_id);
    }
    TrackImage(alias_id);
    return alias_id;
}

void TextureCache::SynchronizeDimensionalAlias(ImageId image_id) {
    auto& destination = slot_images[image_id];
    if (False(destination.flags & ImageFlagBits::DimensionalAliasStale)) {
        return;
    }

    ImageId source_id{};
    // Resolve against the live page table. Overlap handling may have expanded and retired entries
    // from the snapshot FindImage started with.
    ForEachImageInRegion(destination.info.guest_address, destination.info.guest_size,
                         [&](ImageId candidate_id, Image& candidate) {
                             if (source_id || candidate_id == image_id ||
                                 !IsDimensionalAlias(destination.info, candidate.info) ||
                                 True(candidate.flags & ImageFlagBits::DimensionalAliasStale)) {
                                 return;
                             }
                             source_id = candidate_id;
                         });

    if (!source_id) {
        // An authoritative peer can disappear only after its contents have been returned to guest
        // memory. Force a full guest refresh instead of ever sampling a known-stale host image.
        destination.flags &= ~(ImageFlagBits::GpuModified | ImageFlagBits::DimensionalAliasStale);
        destination.flags |= ImageFlagBits::CpuDirty;
        return;
    }

    auto& source = slot_images[source_id];
    RefreshImage(source);
    destination.CopyImage(source);
    destination.flags &=
        ~(ImageFlagBits::Dirty | ImageFlagBits::GpuModified | ImageFlagBits::DimensionalAliasStale);
    destination.flags |= source.flags & ImageFlagBits::GpuModified;
    destination.mip_hashes = source.mip_hashes;
}

void TextureCache::MarkDimensionalAliasesStale(ImageId authoritative_id) {
    std::scoped_lock lock{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    auto& authoritative = slot_images[authoritative_id];
    authoritative.flags &= ~ImageFlagBits::DimensionalAliasStale;

    ForEachImageInRegion(authoritative.info.guest_address, authoritative.info.guest_size,
                         [&](ImageId image_id, Image& image) {
                             if (image_id == authoritative_id ||
                                 !IsDimensionalAlias(authoritative.info, image.info)) {
                                 return;
                             }
                             // Cancel a queued stale readback before it can overwrite guest memory.
                             image.flags &= ~ImageFlagBits::GcPending;
                             image.flags |= ImageFlagBits::DimensionalAliasStale;
                             image.InvalidateTexelBufferSync();
                         });
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};

    // A volume created with VK_EXT_image_2d_view_of_3d can directly back compatible writable
    // 2D slice views. If the array interpretation was discovered first, promote it once to the
    // volume master. Thereafter all bindings resolve to that one ImageId, allowing the existing
    // attachment-feedback-loop path to see the real alias and eliminating per-draw copies.
    if (!image_id && instance.Is2dViewOf3dSupported()) {
        for (const auto cache_id : image_ids) {
            auto& candidate = slot_images[cache_id];
            if (True(candidate.flags & ImageFlagBits::DimensionalAliasStale) ||
                !IsDimensionalAlias(info, candidate.info)) {
                continue;
            }
            if (info.type == AmdGpu::ImageType::Color3D &&
                candidate.info.type != AmdGpu::ImageType::Color3D) {
                image_id = CreateDimensionalAlias(info, cache_id);
                break;
            }
            if (CanUseVolumeMaster(candidate.info, info)) {
                image_id = cache_id;
                break;
            }
        }
    }

    if (!image_id) {
        // Resolve same-dimensional entries first. For example, grow an old 63-layer 2D view to a
        // 64-layer view before retaining the separate 3D peer. This prevents duplicate host
        // images of the requested type and gives stale-state propagation a single lineage.
        ImageIds ordered_image_ids;
        for (const auto cache_id : image_ids) {
            if (slot_images[cache_id].info.type == info.type) {
                ordered_image_ids.push_back(cache_id);
            }
        }
        for (const auto cache_id : image_ids) {
            if (slot_images[cache_id].info.type != info.type) {
                ordered_image_ids.push_back(cache_id);
            }
        }

        for (const auto cache_id : ordered_image_ids) {
            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        SynchronizeDimensionalAlias(image_id);
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
            view_mip = -1;
            view_slice = -1;
        } else if (!image_resolved.info.resources.Contains(info.resources) &&
                   !CanUseVolumeMaster(image_resolved.info, info)) {
            // The overlap search picked an image which cannot back the requested view. Never
            // discard it here: it may contain GPU-only results needed by a later aliasing view.
            LOG_WARNING(Render_Vulkan,
                        "Image overlap resolve selected an undersized resource:\n"
                        "  requested addr={:#x} size={:#x} type={} format={} M:{} L:{} tile={}\n"
                        "  resolved  addr={:#x} size={:#x} type={} format={} M:{} L:{} tile={} "
                        "flags={:#x} target={} bound={}",
                        info.guest_address, info.guest_size, AmdGpu::NameOf(info.type),
                        vk::to_string(info.pixel_format), info.resources.levels,
                        info.resources.layers, AmdGpu::NameOf(info.tile_mode),
                        image_resolved.info.guest_address, image_resolved.info.guest_size,
                        AmdGpu::NameOf(image_resolved.info.type),
                        vk::to_string(image_resolved.info.pixel_format),
                        image_resolved.info.resources.levels, image_resolved.info.resources.layers,
                        AmdGpu::NameOf(image_resolved.info.tile_mode),
                        static_cast<u32>(image_resolved.flags),
                        static_cast<u32>(image_resolved.binding.is_target),
                        static_cast<u32>(image_resolved.binding.is_bound));

            const bool can_expand =
                image_resolved.info.guest_address == info.guest_address &&
                image_resolved.info.type == info.type &&
                image_resolved.info.tile_mode == info.tile_mode &&
                image_resolved.info.BlockDim() == info.BlockDim() &&
                image_resolved.info.props.is_block == info.props.is_block &&
                image_resolved.info.props.is_depth == info.props.is_depth &&
                image_resolved.info.num_bits == info.num_bits &&
                image_resolved.info.num_samples == info.num_samples &&
                IsVulkanFormatCompatible(image_resolved.info.pixel_format, info.pixel_format);
            if (can_expand) {
                auto expanded_info = info;
                const auto expanded_resources =
                    info.resources.ExpandedWith(image_resolved.info.resources);
                if (expanded_resources != info.resources) {
                    // Do not synthesize a larger guest allocation from the host resource union.
                    expanded_info.ExpandResources(expanded_resources);
                }
                image_id = ExpandImage(expanded_info, image_id);
            } else {
                // Keep the selected alias alive and allocate a distinct backing image below.
                image_id = {};
            }
            view_mip = -1;
            view_slice = -1;
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    if (desc.type == BindingType::Storage) {
        // RefreshImage may establish that the raw texel buffer and image are coherent. A storage
        // binding is about to write the image, so invalidate that lineage only after the refresh.
        MarkDimensionalAliasesStale(image_id);
        image.MarkGpuModified();
    }
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{MetaDataInfo::Type::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{MetaDataInfo::Type::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    MarkDimensionalAliasesStale(image_id);
    image.MarkGpuModified();
    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(
            desc.info.meta_info.htile_addr,
            MetaDataInfo{MetaDataInfo::Type::HTile, image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    MarkDimensionalAliasesStale(image_id);
    image.MarkGpuModified();
    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    const Core::PerfTelemetry::ScopedGpuWaitContext refresh_context{
        Core::PerfTelemetry::GpuWaitContext::ImageRefresh, image.info.guest_address,
        image.info.guest_size};

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.guest_resources.layers;
    const u32 num_mips = image.info.guest_resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.guest_mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    if (auto barrier = in_buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
    }

    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    image.Upload(image_copies, buffer, offset);
    if (is_gpu_dirty) {
        // GpuDirty guarantees ObtainBufferForImage selected the persistent buffer-cache copy.
        // Detiling uploaded that exact raw guest data into the image, so a following formatted
        // read-modify-write binding must not tile the unchanged image straight back into it.
        image.MarkTexelBufferSynced(in_buffer->Handle(), in_offset, image.info.guest_size);
    }
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    std::scoped_lock lock{mutex};
    const Core::PerfTelemetry::ScopedGpuWaitContext lock_context{
        Core::PerfTelemetry::GpuWaitContext::TextureCacheLock};
    bool pressured = false;
    bool aggressive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;
    constexpr u64 MaxDownloadBytesPerPass = 32ULL * 1024 * 1024;
    u64 download_bytes = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggressive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggressive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggressive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        auto& image = slot_images[image_id];
        if (True(image.flags & ImageFlagBits::GcPending)) {
            return false;
        }
        const bool download = image.SafeToDownload();
        if (download) {
            // A dedicated host-visible buffer stays alive until its deferred callback completes.
            // Keep only one such snapshot in flight so GC cannot replace a VRAM problem with
            // unbounded WDDM/host commit while the GPU is behind.
            if (pending_gc_download_bytes != 0) {
                return true;
            }
            const u32 image_size = image.info.guest_size;
            // Do not turn one pressured frame into an unbounded GPU-to-CPU transfer. Always allow
            // the first image (even if it is larger than the ring); subsequent safe images wait
            // for the next GC pass once the bounded budget would be exceeded.
            if (download_bytes != 0 &&
                image_size >
                    MaxDownloadBytesPerPass - std::min(download_bytes, MaxDownloadBytesPerPass)) {
                return true;
            }
            buffer_cache.ReadEdgeImagePages(image);
            image.flags |= ImageFlagBits::GcPending;
            pending_gc_download_bytes += image_size;
            DownloadImageMemory(image_id, false, true);
            download_bytes += image_size;
        } else {
            FreeImage(image_id);
        }
        --num_deletions;
        if (total_used_memory < critical_gc_memory) {
            if (aggressive) {
                num_deletions >>= 2;
                aggressive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Retire only old entries and let normal queue submission release them. A garbage-collection
    // pass must never turn every guest frame into a full scheduler drain.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(Image& image) {
    // A resource used again before its readback callback completes remains authoritative on the
    // GPU. The queued callback observes this cancellation and neither overwrites guest memory nor
    // retires the live image.
    image.flags &= ~ImageFlagBits::GcPending;
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
