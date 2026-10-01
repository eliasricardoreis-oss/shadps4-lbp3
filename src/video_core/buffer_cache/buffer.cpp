// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/performance_telemetry.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <limits>
#include <vk_mem_alloc.h>

namespace VideoCore {

std::string_view BufferTypeName(MemoryUsage type) {
    switch (type) {
    case MemoryUsage::Upload:
        return "Upload";
    case MemoryUsage::Download:
        return "Download";
    case MemoryUsage::Stream:
        return "Stream";
    case MemoryUsage::DeviceLocal:
        return "DeviceLocal";
    case MemoryUsage::HostImported:
        return "HostImported";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
    return usage != MemoryUsage::DeviceLocal && usage != MemoryUsage::HostImported
               ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
               : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::Upload:
    case MemoryUsage::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryUsage::Download:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::HostImported:
        return {};
    }
    return {};
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::HostImported:
    case MemoryUsage::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::Upload:
    case MemoryUsage::Download:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    Destroy();
}

void UniqueBuffer::Destroy() noexcept {
    if (!buffer) {
        return;
    }
    if (IsHostImported()) {
        const vk::DeviceMemory imported_memory{
            std::bit_cast<VkDeviceMemory>(allocation),
        };
        device.destroyBuffer(buffer);
        device.freeMemory(imported_memory);
    } else {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
    allocator = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    buffer = VK_NULL_HANDLE;
    bda_addr = 0;
}

void UniqueBuffer::Create(const vk::BufferCreateInfo& buffer_ci, MemoryUsage usage,
                          VmaAllocationInfo* out_alloc_info) {
    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    const VmaAllocationCreateFlags bda_flag =
        with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    VkBuffer unsafe_buffer{};
    VkResult result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                      &allocation, out_alloc_info);
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer with error {}",
               vk::to_string(vk::Result{result}));
    buffer = vk::Buffer{unsafe_buffer};

    if (with_bda) {
        vk::BufferDeviceAddressInfo bda_info{
            .buffer = buffer,
        };
        auto bda_result = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_result != 0, "Failed to get buffer device address");
        bda_addr = bda_result;
    }
}

bool UniqueBuffer::CreateHostImported(const Vulkan::Instance& instance,
                                      const vk::BufferCreateInfo& buffer_ci, VAddr cpu_addr,
                                      u64 size_bytes) {
    if (!instance.SupportsExternalMemoryHost() || buffer || cpu_addr == 0 || size_bytes == 0) {
        return false;
    }
    const vk::DeviceSize host_alignment = instance.MinImportedHostPointerAlignment();
    if (host_alignment == 0 || cpu_addr % host_alignment != 0 || size_bytes % host_alignment != 0) {
        return false;
    }

    void* const host_pointer = std::bit_cast<void*>(cpu_addr);
    const auto [host_result, host_properties] = device.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, host_pointer);
    if (host_result != vk::Result::eSuccess || host_properties.memoryTypeBits == 0) {
        return false;
    }

    const vk::ExternalMemoryBufferCreateInfo external_ci = {
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
    };
    vk::BufferCreateInfo imported_buffer_ci = buffer_ci;
    imported_buffer_ci.pNext = &external_ci;
    const auto [create_result, imported_buffer] = device.createBuffer(imported_buffer_ci);
    if (create_result != vk::Result::eSuccess) {
        return false;
    }

    const vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(imported_buffer);
    if (requirements.size > size_bytes || requirements.size % host_alignment != 0 ||
        cpu_addr % requirements.alignment != 0) {
        device.destroyBuffer(imported_buffer);
        return false;
    }

    const u32 compatible_types = requirements.memoryTypeBits & host_properties.memoryTypeBits;
    const auto& memory_properties = instance.GetMemoryProperties();
    constexpr vk::MemoryPropertyFlags required_flags =
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    u32 selected_type = std::numeric_limits<u32>::max();
    int selected_score = -1;
    for (u32 index = 0; index < memory_properties.memoryTypeCount; ++index) {
        if ((compatible_types & (1U << index)) == 0) {
            continue;
        }
        const vk::MemoryPropertyFlags flags = memory_properties.memoryTypes[index].propertyFlags;
        if ((flags & required_flags) != required_flags) {
            continue;
        }
        const int score = (bool(flags & vk::MemoryPropertyFlagBits::eDeviceLocal) ? 2 : 0) +
                          (bool(flags & vk::MemoryPropertyFlagBits::eHostCached) ? 1 : 0);
        if (score > selected_score) {
            selected_type = index;
            selected_score = score;
        }
    }
    if (selected_type == std::numeric_limits<u32>::max()) {
        device.destroyBuffer(imported_buffer);
        return false;
    }

    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    const vk::MemoryDedicatedAllocateInfo dedicated_info = {
        .buffer = imported_buffer,
    };
    const vk::MemoryAllocateFlagsInfo flags_info = {
        .pNext = &dedicated_info,
        .flags = with_bda ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{},
    };
    const vk::ImportMemoryHostPointerInfoEXT import_info = {
        .pNext = with_bda ? static_cast<const void*>(&flags_info)
                          : static_cast<const void*>(&dedicated_info),
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        .pHostPointer = host_pointer,
    };
    const vk::MemoryAllocateInfo allocate_info = {
        .pNext = &import_info,
        .allocationSize = requirements.size,
        .memoryTypeIndex = selected_type,
    };
    const auto [allocate_result, imported_memory] = device.allocateMemory(allocate_info);
    if (allocate_result != vk::Result::eSuccess) {
        device.destroyBuffer(imported_buffer);
        return false;
    }
    const vk::Result bind_result = device.bindBufferMemory(imported_buffer, imported_memory, 0);
    if (bind_result != vk::Result::eSuccess) {
        device.freeMemory(imported_memory);
        device.destroyBuffer(imported_buffer);
        return false;
    }

    vk::DeviceAddress imported_bda = 0;
    if (with_bda) {
        imported_bda = device.getBufferAddress(vk::BufferDeviceAddressInfo{
            .buffer = imported_buffer,
        });
        if (imported_bda == 0) {
            device.destroyBuffer(imported_buffer);
            device.freeMemory(imported_memory);
            return false;
        }
    }

    allocator = VK_NULL_HANDLE;
    allocation = std::bit_cast<VmaAllocation>(static_cast<VkDeviceMemory>(imported_memory));
    buffer = imported_buffer;
    bda_addr = imported_bda;
    return true;
}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
               VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // Create buffer object.
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
    };
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, usage, &alloc_info);

    const auto device = instance->GetDevice();
    Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);

    // Map it if it is host visible.
    VkMemoryPropertyFlags property_flags{};
    vmaGetAllocationMemoryProperties(instance->GetAllocator(), buffer.allocation, &property_flags);
    if (alloc_info.pMappedData) {
        mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
    }
    is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

std::optional<UniqueBuffer> Buffer::TryImportHostMemory(vk::BufferUsageFlags flags) {
    if (usage != MemoryUsage::DeviceLocal || cpu_addr == 0 || size_bytes == 0) {
        return std::nullopt;
    }
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
    };
    UniqueBuffer imported{instance->GetDevice(), instance->GetAllocator()};
    if (!imported.CreateHostImported(*instance, buffer_ci, cpu_addr, size_bytes)) {
        return std::nullopt;
    }

    std::swap(buffer.allocator, imported.allocator);
    std::swap(buffer.allocation, imported.allocation);
    std::swap(buffer.buffer, imported.buffer);
    std::swap(buffer.bda_addr, imported.bda_addr);
    usage = MemoryUsage::HostImported;
    is_coherent = true;
    mapped_data = {};
    MarkHostWrite();
    Vulkan::SetObjectName(instance->GetDevice(), Handle(), "HostImported Buffer {:#x}:{:#x}",
                          cpu_addr, size_bytes);
    return std::optional<UniqueBuffer>{std::move(imported)};
}

void Buffer::Fill(u64 offset, u64 num_bytes, u32 value) {
    scheduler->EndRendering();
    ASSERT_MSG(offset % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer size must be a multiple of 4 bytes");
    const auto cmdbuf = scheduler->CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.fillBuffer(buffer, offset, num_bytes, value);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x4000;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x1000;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           MemoryUsage usage, u64 size_bytes,
                           Core::PerfTelemetry::GpuWaitResource telemetry_resource_)
    : Buffer{instance, scheduler, usage, 0, AllFlags, size_bytes},
      telemetry_resource{telemetry_resource_} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, Handle(), "StreamBuffer({}):{:#x}", BufferTypeName(usage),
                          size_bytes);
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (mapped_data.empty()) {
        return {nullptr, 0};
    }
    if (!is_coherent && usage == MemoryUsage::Stream) {
        size = Common::AlignUp(size, instance->NonCoherentAtomSize());
    }

    const auto reserved_offset = Reserve(size, alignment, allow_wait);
    if (!reserved_offset) {
        return {nullptr, 0};
    }
    return {mapped_data.data() + *reserved_offset, *reserved_offset};
}

std::optional<u64> StreamBuffer::Reserve(u64 size, u64 alignment, bool allow_wait) {
    if (size > this->size_bytes) {
        return std::nullopt;
    }

    mapped_size = size;

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the
        // state.
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    const u64 mapped_upper_bound = offset + size;
    if (!WaitPendingOperations(mapped_upper_bound, allow_wait)) {
        return std::nullopt;
    }

    return offset;
}

void StreamBuffer::Commit(bool flush_host_writes) {
    if (flush_host_writes && !is_coherent) {
        if (usage == MemoryUsage::Download) {
            vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, offset,
                                    mapped_size);
        } else {
            vmaFlushAllocation(instance->GetAllocator(), buffer.allocation, offset, mapped_size);
        }
    }

    offset += mapped_size;
    if (current_watch_cursor != 0 &&
        current_watches[current_watch_cursor - 1].tick == scheduler->CurrentTick()) {
        current_watches[current_watch_cursor - 1].upper_bound = offset;
        return;
    }

    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = scheduler->CurrentTick();
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

bool StreamBuffer::WaitPendingOperations(u64 requested_upper_bound, bool allow_wait) {
    if (!invalidation_mark) {
        return true;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        if (!scheduler->IsFree(watch.tick) && !allow_wait) {
            return false;
        }
        scheduler->Wait(watch.tick, {.source = Core::PerfTelemetry::GpuWaitSource::StreamReuse,
                                     .resource = telemetry_resource,
                                     .resource_id = reinterpret_cast<u64>(this),
                                     .capacity_bytes = size_bytes,
                                     .request_bytes = mapped_size,
                                     .offset_bytes = offset});
        // Download-buffer callbacks consume the mapped bytes on the CPU after the
        // GPU reaches this tick. Run those callbacks before the ring is allowed to
        // reuse the same range. Waiting on the semaphore alone only makes the
        // transfer complete; it does not drain Scheduler::pending_ops.
        if (usage == MemoryUsage::Download) {
            scheduler->PopPendingOperations();
        }
        wait_bound = watch.upper_bound;
        ++wait_cursor;
    }
    return true;
}

GpuScratchBuffer::GpuScratchBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                                   u64 size_bytes)
    : scheduler{scheduler_},
      buffer{instance, scheduler_, MemoryUsage::DeviceLocal, 0, AllFlags, size_bytes} {}

std::optional<u64> GpuScratchBuffer::Reserve(u64 size, u64 alignment) {
    if (size == 0 || size > SizeBytes()) {
        return std::nullopt;
    }

    u64 offset = alignment ? Common::AlignUp(cursor, alignment) : cursor;
    if (offset > SizeBytes() - size) {
        offset = 0;
        wrapped = true;
    }

    if (wrapped) {
        // Earlier detile->image-copy / image-copy->tile uses are already recorded on
        // this queue, possibly in earlier submissions. Order their reads AND writes
        // before this range is overwritten. In particular, a compute write must not
        // overtake the transfer read that still consumes a previous detile result.
        // This is GPU-only storage: no host data or callback needs a completed tick.
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 reuse_barrier{
            .srcStageMask =
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask =
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer.Handle(),
            .offset = offset,
            .size = size,
        };
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &reuse_barrier,
        });
        Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::TileScratchReuseBarriers);
    }

    cursor = offset + size;
    Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::TileScratchReservations);
    Core::PerfTelemetry::Increment(Core::PerfTelemetry::Counter::TileScratchBytes, size);
    return offset;
}

} // namespace VideoCore
