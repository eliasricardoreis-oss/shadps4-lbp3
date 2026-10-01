// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/gpu_wait_log.h"
#include "core/performance_telemetry.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_info.h>
#include <mach/thread_status.h>
#include <pthread.h>
#endif

#include "common/elf_info.h"
#include "common/guest_time_stall.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/thread.h"

namespace Core::PerfTelemetry {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t CounterCount = static_cast<size_t>(Counter::Count);
constexpr size_t TimeMetricCount = static_cast<size_t>(TimeMetric::Count);
std::atomic_bool start_requested{};
std::filesystem::path output_directory;

struct ProcessSnapshot {
    u64 cpu_ns{};
    u64 rss_bytes{};
};

[[nodiscard]] ProcessSnapshot GetProcessSnapshot() {
    ProcessSnapshot result{};
#ifndef _WIN32
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        const auto timeval_to_ns = [](const timeval& value) {
            return static_cast<u64>(value.tv_sec) * 1'000'000'000ULL +
                   static_cast<u64>(value.tv_usec) * 1'000ULL;
        };
        result.cpu_ns = timeval_to_ns(usage.ru_utime) + timeval_to_ns(usage.ru_stime);
#ifdef __APPLE__
        mach_task_basic_info_data_t task_info_data{};
        mach_msg_type_number_t task_info_count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&task_info_data),
                      &task_info_count) == KERN_SUCCESS) {
            result.rss_bytes = task_info_data.resident_size;
        }
#else
        result.rss_bytes = static_cast<u64>(usage.ru_maxrss) * 1024ULL;
#endif
    }
#endif
    return result;
}

[[nodiscard]] std::string CsvEscape(std::string_view value) {
    bool requires_quotes = false;
    for (const char c : value) {
        requires_quotes |= c == ',' || c == '"' || c == '\n' || c == '\r';
    }
    if (!requires_quotes) {
        return std::string{value};
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char c : value) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

[[nodiscard]] std::string MakeTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif
    std::ostringstream out;
    out << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0')
        << millis;
    return out.str();
}

#ifdef __APPLE__
struct SampleKey {
    u64 thread_id{};
    u64 pc{};

    bool operator==(const SampleKey&) const = default;
};

struct SampleKeyHash {
    size_t operator()(const SampleKey& value) const noexcept {
        return std::hash<u64>{}(value.pc ^ (value.thread_id * 0x9e3779b97f4a7c15ULL));
    }
};

struct ThreadSample {
    std::string name;
    float cpu_percent{};
    integer_t run_state{};
};
#endif

class Recorder {
public:
    ~Recorder() {
        Stop();
    }

    void Start() {
        std::scoped_lock lock{lifecycle_mutex};
        if (enabled.load(std::memory_order_acquire)) {
            return;
        }

        const auto telemetry_dir =
            output_directory.empty()
                ? Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "telemetry"
                : output_directory;
        std::filesystem::create_directories(telemetry_dir);
        const auto stem = telemetry_dir / (std::string{Common::ElfInfo::Instance().GameSerial()} +
                                           '_' + MakeTimestamp());
        frame_path = stem.string() + "_frames.csv";
        sample_path = stem.string() + "_samples.csv";
        thread_path = stem.string() + "_threads.csv";
        meta_path = stem.string() + "_meta.txt";

        frames.open(frame_path, std::ios::out | std::ios::trunc);
        samples.open(sample_path, std::ios::out | std::ios::trunc);
        threads.open(thread_path, std::ios::out | std::ios::trunc);
        metadata.open(meta_path, std::ios::out | std::ios::trunc);
        if (!frames.is_open() || !samples.is_open() || !threads.is_open() || !metadata.is_open()) {
            LOG_ERROR(Core, "Could not open performance telemetry files in {}",
                      telemetry_dir.string());
            return;
        }

        frames << "frame,elapsed_ms,frame_ms,fps,process_cpu_ms,process_cpu_percent,rss_mb,"
                  "game_width,game_height,output_width,output_height,pending_flips,request_depth,"
                  "gpu_frame_cpu_ms,draw_cpu_ms,dispatch_cpu_ms,resource_bind_cpu_ms,"
                  "on_submit_cpu_ms,raster_flush_cpu_ms,prepare_frame_cpu_ms,present_cpu_ms,"
                  "frame_pool_wait_ms,present_fence_wait_ms,vk_submit_cpu_ms,gpu_wait_ms,"
                  "gfx_pipeline_compile_ms,compute_pipeline_compile_ms,guest_shader_compile_ms,"
                  "host_shader_compile_ms,fault_service_ms,sampler_overhead_ms,draw_calls,"
                  "draws_emitted,indirect_draw_calls,dispatch_calls,dispatches_emitted,"
                  "descriptor_writes,render_pass_begins,render_pass_ends,vk_submits,gpu_waits,"
                  "gpu_frames,gfx_submits,asc_submits,gfx_dwords,asc_dwords,"
                  "gfx_pipeline_compiles,compute_pipeline_compiles,guest_shader_compiles,"
                  "host_shader_compiles,guest_write_faults,guest_read_faults,present_calls,"
                  "direct_import_attempts,direct_import_successes,direct_import_failures,"
                  "direct_buffer_binds,direct_readback_bytes,direct_upload_bytes,"
                  "direct_fault_submits,direct_fault_finishes,direct_visibility_barriers,"
                  "ordered_guest_releases,guest_fence_submits,lbp3_ng_cpu_hle_dispatches,"
                  "lbp3_ng_cpu_hle_releases,"
                  "guest_stall_ms,guest_stall_active,submit_mutex_wait_ms,queue_submit_cpu_ms,"
                  "command_pool_wait_ms,command_buffer_acquire_cpu_ms,submit_pending_ops_cpu_ms,"
                  "vk_empty_submits,command_buffer_acquires,"
                  "reused_presents,empty_flip_slots,swapchain_acquire_cpu_ms,"
                  "swapchain_present_cpu_ms,reuse_prepare_cpu_ms,reuse_fence_wait_ms,"
                  "gnm_submit_wait_ms,flip_timing_valid,flip_prepare_start_ms,flip_ready_ms,"
                  "flip_present_start_ms,flip_present_end_ms,flip_prepare_cpu_ms,"
                  "flip_queue_wait_ms,flip_present_cpu_ms,flip_signal_cpu_ms,reuse_skipped_busy,"
                  "videoout_flip_rate,videoout_vblank_count,flip_feedback_end_ms,"
                  "texture_fault_lock_wait_ms,buffer_fault_lock_wait_ms,"
                  "texture_fault_lock_waits,buffer_fault_lock_waits,"
                  "tile_scratch_reservations,tile_scratch_bytes,tile_scratch_reuse_barriers,"
                  "indexed_quad_draws,expanded_indexed_quad_draws,indexed_quad_fallbacks,"
                  "expanded_quad_index_bytes,quad_index_expand_cpu_ms\n";
        samples << "elapsed_ms,thread_id,thread_name,kind,pc,image,image_offset,symbol,"
                   "symbol_offset,samples\n";
        threads << "elapsed_ms,thread_id,thread_name,cpu_percent,run_state\n";
        metadata << "serial=" << Common::ElfInfo::Instance().GameSerial() << '\n';
        metadata << "flip_timing_version=1\n"
                    "flip_timing_clock=steady_clock relative to start_steady_ns; -1 if invalid\n"
                    "flip_ready_semantics=host request ready, not GPU completion\n"
                    "flip_present_end_semantics=host present returned, not display scanout\n"
                    "videoout_rate_version=1\n"
                    "videoout_flip_rate_semantics=guest requested extra vblanks per flip\n"
                    "flip_feedback_version=1\n"
                    "flip_feedback_end_semantics=guest events and previous output label published\n"
                    "flip_signal_cpu_ms_semantics=CPU tail after host Present returns\n";
        metadata << "eboot_base=0x" << std::hex << MemoryPatcher::g_eboot_address << '\n';
        metadata << "eboot_size=0x" << MemoryPatcher::g_eboot_image_size << std::dec << '\n';
        metadata << "frames=" << frame_path.string() << '\n';
        metadata << "samples=" << sample_path.string() << '\n';
        metadata << "threads=" << thread_path.string() << '\n';
        metadata.flush();

        for (auto& counter : counters) {
            counter.store(0, std::memory_order_relaxed);
        }
        for (auto& timing : timings) {
            timing.store(0, std::memory_order_relaxed);
        }

        frame_number = 0;
        start_time = Clock::now();
        metadata << "start_unix_ns="
                 << std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count()
                 << '\n';
        metadata << "start_steady_ns="
                 << std::chrono::duration_cast<std::chrono::nanoseconds>(
                        start_time.time_since_epoch())
                        .count()
                 << '\n';
#ifdef __APPLE__
        mach_timebase_info_data_t timebase{};
        mach_timebase_info(&timebase);
        metadata << "start_mach_absolute_ticks=" << mach_absolute_time() << '\n';
        metadata << "mach_timebase_numer=" << timebase.numer << '\n';
        metadata << "mach_timebase_denom=" << timebase.denom << '\n';
        metadata << "pid=" << getpid() << '\n';
#endif
        metadata.flush();
        const auto gpu_wait_path = std::filesystem::path{stem.string() + "_gpu_waits.csv"};
        const bool gpu_wait_open = gpu_wait_log.Open(gpu_wait_path, start_time);
        metadata
            << "gpu_wait_attribution_version=1\ngpu_wait_log_open=" << gpu_wait_open << '\n'
            << "gpu_waits=" << gpu_wait_path.string() << '\n'
            << "gpu_wait_clock=steady_clock relative to start_steady_ns\n"
               "gpu_wait_semantics=blocking semaphore fallback only; includes final tick refresh\n"
               "gpu_wait_context_bits=1:texture_cache_lock,2:image_refresh,4:buffer_upload,"
               "8:upload_tracker_locks,16:image_download\n"
               "gpu_wait_retention=two rotating CSV files, at most 4 MiB each; latest events\n"
               "fault_lock_wait_semantics=wall time of contended acquisitions, summed across "
               "threads\n";
        metadata.flush();
        last_frame_time = start_time;
        last_frame_flush = start_time;
        last_process = GetProcessSnapshot();
        const auto guest_stall = Common::GetGuestTimeStallTracker().GetSnapshot();
        last_guest_stall = guest_stall.elapsed;
        enabled.store(true, std::memory_order_release);

#ifdef __APPLE__
        sampler = std::jthread([this](std::stop_token token) { SampleThreads(token); });
#endif

        LOG_INFO(Core, "Performance telemetry recording to {}", frame_path.string());
    }

    void Stop() {
        std::scoped_lock lock{lifecycle_mutex};
        if (!enabled.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        gpu_wait_log.Close();
#ifdef __APPLE__
        sampler.request_stop();
        if (sampler.joinable()) {
            sampler.join();
        }
#endif
        frames.flush();
        samples.flush();
        threads.flush();
        metadata.flush();
        LOG_INFO(Core, "Performance telemetry stopped after {} frames; log {}", frame_number,
                 frame_path.string());
    }

    [[nodiscard]] bool IsEnabled() const noexcept {
        return enabled.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 GetFrameCount() const noexcept {
        return recorded_frames.load(std::memory_order_relaxed);
    }

    void Increment(Counter counter, u64 amount) noexcept {
        if (!IsEnabled()) {
            return;
        }
        counters[static_cast<size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
    }

    void AddTime(TimeMetric metric, std::chrono::nanoseconds duration) noexcept {
        if (!IsEnabled()) {
            return;
        }
        timings[static_cast<size_t>(metric)].fetch_add(duration.count(), std::memory_order_relaxed);
    }

    void RecordGpuWait(const GpuWaitEvent& event) noexcept {
        if (IsEnabled()) {
            gpu_wait_log.Record(event);
        }
    }

    void RecordFrame(u32 pending_flips, u32 request_depth, u32 game_width, u32 game_height,
                     u32 output_width, u32 output_height, const FlipTiming& flip_timing) {
        if (!IsEnabled()) {
            return;
        }

        const auto now = Clock::now();
        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time).count();
        const auto frame_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_frame_time).count();
        last_frame_time = now;

        std::array<u64, CounterCount> frame_counters{};
        std::array<s64, TimeMetricCount> frame_timings{};
        for (size_t i = 0; i < CounterCount; ++i) {
            frame_counters[i] = counters[i].exchange(0, std::memory_order_acq_rel);
        }
        for (size_t i = 0; i < TimeMetricCount; ++i) {
            frame_timings[i] = timings[i].exchange(0, std::memory_order_acq_rel);
        }

        const auto process = GetProcessSnapshot();
        const u64 process_delta_ns = process.cpu_ns - last_process.cpu_ns;
        last_process = process;
        const double frame_ms = static_cast<double>(frame_ns) / 1'000'000.0;
        const double fps = frame_ns > 0 ? 1'000'000'000.0 / static_cast<double>(frame_ns) : 0.0;
        const double process_cpu_percent =
            frame_ns > 0
                ? 100.0 * static_cast<double>(process_delta_ns) / static_cast<double>(frame_ns)
                : 0.0;

        const auto guest_stall = Common::GetGuestTimeStallTracker().GetSnapshot();
        const auto guest_stall_delta = guest_stall.elapsed - last_guest_stall;
        last_guest_stall = guest_stall.elapsed;

        const auto counter = [&](Counter value) {
            return frame_counters[static_cast<size_t>(value)];
        };
        const auto milliseconds = [&](TimeMetric value) {
            return static_cast<double>(frame_timings[static_cast<size_t>(value)]) / 1'000'000.0;
        };
        const bool flip_timing_valid = flip_timing.prepare_begin >= start_time &&
                                       flip_timing.ready >= flip_timing.prepare_begin &&
                                       flip_timing.present_begin >= flip_timing.ready &&
                                       flip_timing.present_end >= flip_timing.present_begin &&
                                       now >= flip_timing.present_end;
        const auto flip_ms = [&](Clock::time_point begin, Clock::time_point end) {
            return flip_timing_valid
                       ? std::chrono::duration<double, std::milli>(end - begin).count()
                       : -1.0;
        };

        ++frame_number;
        recorded_frames.store(frame_number, std::memory_order_relaxed);
        frames << std::fixed << std::setprecision(3) << frame_number << ','
               << static_cast<double>(elapsed_ns) / 1'000'000.0 << ',' << frame_ms << ',' << fps
               << ',' << static_cast<double>(process_delta_ns) / 1'000'000.0 << ','
               << process_cpu_percent << ','
               << static_cast<double>(process.rss_bytes) / (1024.0 * 1024.0) << ',' << game_width
               << ',' << game_height << ',' << output_width << ',' << output_height << ','
               << pending_flips << ',' << request_depth << ','
               << milliseconds(TimeMetric::GpuFrameCpu) << ',' << milliseconds(TimeMetric::DrawCpu)
               << ',' << milliseconds(TimeMetric::DispatchCpu) << ','
               << milliseconds(TimeMetric::ResourceBindCpu) << ','
               << milliseconds(TimeMetric::OnSubmitCpu) << ','
               << milliseconds(TimeMetric::RasterFlushCpu) << ','
               << milliseconds(TimeMetric::PrepareFrameCpu) << ','
               << milliseconds(TimeMetric::PresentCpu) << ','
               << milliseconds(TimeMetric::FramePoolWait) << ','
               << milliseconds(TimeMetric::PresentFenceWait) << ','
               << milliseconds(TimeMetric::VkSubmitCpu) << ',' << milliseconds(TimeMetric::GpuWait)
               << ',' << milliseconds(TimeMetric::GraphicsPipelineCompile) << ','
               << milliseconds(TimeMetric::ComputePipelineCompile) << ','
               << milliseconds(TimeMetric::GuestShaderCompile) << ','
               << milliseconds(TimeMetric::HostShaderCompile) << ','
               << milliseconds(TimeMetric::FaultService) << ','
               << milliseconds(TimeMetric::SamplerOverhead) << ',' << counter(Counter::DrawCalls)
               << ',' << counter(Counter::DrawsEmitted) << ','
               << counter(Counter::IndirectDrawCalls) << ',' << counter(Counter::DispatchCalls)
               << ',' << counter(Counter::DispatchesEmitted) << ','
               << counter(Counter::DescriptorWrites) << ',' << counter(Counter::RenderPassBegins)
               << ',' << counter(Counter::RenderPassEnds) << ',' << counter(Counter::VkSubmits)
               << ',' << counter(Counter::GpuWaits) << ',' << counter(Counter::GpuFrames) << ','
               << counter(Counter::GfxSubmits) << ',' << counter(Counter::AscSubmits) << ','
               << counter(Counter::GfxDwords) << ',' << counter(Counter::AscDwords) << ','
               << counter(Counter::GraphicsPipelineCompiles) << ','
               << counter(Counter::ComputePipelineCompiles) << ','
               << counter(Counter::GuestShaderCompiles) << ','
               << counter(Counter::HostShaderCompiles) << ',' << counter(Counter::GuestWriteFaults)
               << ',' << counter(Counter::GuestReadFaults) << ',' << counter(Counter::PresentCalls)
               << ',' << counter(Counter::DirectImportAttempts) << ','
               << counter(Counter::DirectImportSuccesses) << ','
               << counter(Counter::DirectImportFailures) << ','
               << counter(Counter::DirectBufferBinds) << ','
               << counter(Counter::DirectReadbackBytes) << ','
               << counter(Counter::DirectUploadBytes) << ',' << counter(Counter::DirectFaultSubmits)
               << ',' << counter(Counter::DirectFaultFinishes) << ','
               << counter(Counter::DirectVisibilityBarriers) << ','
               << counter(Counter::OrderedGuestReleases) << ','
               << counter(Counter::GuestFenceSubmits) << ','
               << counter(Counter::Lbp3NgCpuHleDispatches) << ','
               << counter(Counter::Lbp3NgCpuHleReleases) << ','
               << std::chrono::duration<double, std::milli>(guest_stall_delta).count() << ','
               << (guest_stall.active ? 1 : 0) << ',' << milliseconds(TimeMetric::SubmitMutexWait)
               << ',' << milliseconds(TimeMetric::QueueSubmitCpu) << ','
               << milliseconds(TimeMetric::CommandPoolWait) << ','
               << milliseconds(TimeMetric::CommandBufferAcquireCpu) << ','
               << milliseconds(TimeMetric::SubmitPendingOpsCpu) << ','
               << counter(Counter::VkEmptySubmits) << ',' << counter(Counter::CommandBufferAcquires)
               << ',' << counter(Counter::ReusedPresents) << ',' << counter(Counter::EmptyFlipSlots)
               << ',' << milliseconds(TimeMetric::SwapchainAcquireCpu) << ','
               << milliseconds(TimeMetric::SwapchainPresentCpu) << ','
               << milliseconds(TimeMetric::ReusePrepareCpu) << ','
               << milliseconds(TimeMetric::ReuseFenceWait) << ','
               << milliseconds(TimeMetric::GnmSubmitWait) << ',' << (flip_timing_valid ? 1 : 0)
               << ',' << flip_ms(start_time, flip_timing.prepare_begin) << ','
               << flip_ms(start_time, flip_timing.ready) << ','
               << flip_ms(start_time, flip_timing.present_begin) << ','
               << flip_ms(start_time, flip_timing.present_end) << ','
               << flip_ms(flip_timing.prepare_begin, flip_timing.ready) << ','
               << flip_ms(flip_timing.ready, flip_timing.present_begin) << ','
               << flip_ms(flip_timing.present_begin, flip_timing.present_end) << ','
               << flip_ms(flip_timing.present_end, now) << ',' << counter(Counter::ReuseSkippedBusy)
               << ',' << flip_timing.videoout_flip_rate << ',' << flip_timing.videoout_vblank_count
               << ',' << flip_ms(start_time, flip_timing.feedback_end) << ','
               << milliseconds(TimeMetric::TextureFaultLockWait) << ','
               << milliseconds(TimeMetric::BufferFaultLockWait) << ','
               << counter(Counter::TextureFaultLockWaits) << ','
               << counter(Counter::BufferFaultLockWaits) << ','
               << counter(Counter::TileScratchReservations) << ','
               << counter(Counter::TileScratchBytes) << ','
               << counter(Counter::TileScratchReuseBarriers) << ','
               << counter(Counter::IndexedQuadDraws) << ','
               << counter(Counter::ExpandedIndexedQuadDraws) << ','
               << counter(Counter::IndexedQuadFallbacks) << ','
               << counter(Counter::ExpandedQuadIndexBytes) << ','
               << milliseconds(TimeMetric::QuadIndexExpandCpu) << '\n';

        if (now - last_frame_flush >= std::chrono::seconds(1)) {
            frames.flush();
            last_frame_flush = now;
        }
    }

private:
#ifdef __APPLE__
    void FlushThreadSamples(const Clock::time_point now,
                            const std::unordered_map<SampleKey, u32, SampleKeyHash>& sample_counts,
                            const std::unordered_map<u64, ThreadSample>& thread_samples) {
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        std::vector<std::pair<SampleKey, u32>> sorted_samples(sample_counts.begin(),
                                                              sample_counts.end());
        std::ranges::sort(sorted_samples, {}, &std::pair<SampleKey, u32>::second);
        std::ranges::reverse(sorted_samples);
        std::unordered_map<u64, u32> rows_per_thread;
        u32 rows_written = 0;
        for (const auto& [key, count] : sorted_samples) {
            if (rows_written >= 128 || rows_per_thread[key.thread_id] >= 12) {
                continue;
            }
            ++rows_per_thread[key.thread_id];
            ++rows_written;

            const auto thread_it = thread_samples.find(key.thread_id);
            const std::string thread_name =
                thread_it != thread_samples.end() ? thread_it->second.name : std::string{"unknown"};
            std::string kind{"host"};
            std::string image{"unknown"};
            std::string symbol;
            u64 image_offset{};
            u64 symbol_offset{};
            if (key.pc >= MemoryPatcher::g_eboot_address &&
                key.pc < MemoryPatcher::g_eboot_address + MemoryPatcher::g_eboot_image_size) {
                kind = "guest_eboot";
                image = "eboot.bin";
                image_offset = key.pc - MemoryPatcher::g_eboot_address;
            } else {
                Dl_info info{};
                if (dladdr(reinterpret_cast<void*>(key.pc), &info) != 0) {
                    if (info.dli_fname != nullptr) {
                        image = std::filesystem::path{info.dli_fname}.filename().string();
                    }
                    if (info.dli_fbase != nullptr) {
                        image_offset = key.pc - reinterpret_cast<u64>(info.dli_fbase);
                    }
                    if (info.dli_sname != nullptr) {
                        symbol = info.dli_sname;
                    }
                    if (info.dli_saddr != nullptr) {
                        symbol_offset = key.pc - reinterpret_cast<u64>(info.dli_saddr);
                    }
                }
            }

            samples << elapsed_ms << ',' << key.thread_id << ',' << CsvEscape(thread_name) << ','
                    << kind << ",0x" << std::hex << key.pc << ',' << CsvEscape(image) << ",0x"
                    << image_offset << ',' << CsvEscape(symbol) << ",0x" << symbol_offset
                    << std::dec << ',' << count << '\n';
        }

        std::vector<std::pair<u64, ThreadSample>> sorted_threads(thread_samples.begin(),
                                                                 thread_samples.end());
        std::ranges::sort(sorted_threads, [](const auto& lhs, const auto& rhs) {
            return lhs.second.cpu_percent > rhs.second.cpu_percent;
        });
        u32 thread_rows = 0;
        for (const auto& [thread_id, sample] : sorted_threads) {
            if (thread_rows >= 64) {
                break;
            }
            if (sample.cpu_percent < 0.1f && sample.run_state != TH_STATE_RUNNING) {
                continue;
            }
            ++thread_rows;
            threads << elapsed_ms << ',' << thread_id << ',' << CsvEscape(sample.name) << ','
                    << std::fixed << std::setprecision(2) << sample.cpu_percent << ','
                    << sample.run_state << '\n';
        }
        samples.flush();
        threads.flush();
    }

    void SampleThreads(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:PerfSampler");
        // 25 Hz is enough to identify a hot guest/host function over a gameplay segment while
        // keeping the profiler itself well below one millisecond per rendered frame.
        constexpr auto sample_period = std::chrono::milliseconds(40);
        auto next_sample = Clock::now();
        auto next_flush = next_sample + std::chrono::seconds(1);
        std::unordered_map<SampleKey, u32, SampleKeyHash> sample_counts;
        std::unordered_map<u64, ThreadSample> thread_samples;
        const thread_t own_thread = mach_thread_self();

        while (!token.stop_requested()) {
            next_sample += sample_period;
            const auto sample_begin = Clock::now();
            thread_act_array_t task_thread_list{};
            mach_msg_type_number_t thread_count{};
            if (task_threads(mach_task_self(), &task_thread_list, &thread_count) == KERN_SUCCESS) {
                for (mach_msg_type_number_t i = 0; i < thread_count; ++i) {
                    const thread_t thread = task_thread_list[i];
                    if (thread == own_thread) {
                        mach_port_deallocate(mach_task_self(), thread);
                        continue;
                    }

                    thread_identifier_info_data_t identifier{};
                    mach_msg_type_number_t identifier_count = THREAD_IDENTIFIER_INFO_COUNT;
                    if (thread_info(thread, THREAD_IDENTIFIER_INFO,
                                    reinterpret_cast<thread_info_t>(&identifier),
                                    &identifier_count) != KERN_SUCCESS) {
                        mach_port_deallocate(mach_task_self(), thread);
                        continue;
                    }

                    thread_basic_info_data_t basic{};
                    mach_msg_type_number_t basic_count = THREAD_BASIC_INFO_COUNT;
                    if (thread_info(thread, THREAD_BASIC_INFO,
                                    reinterpret_cast<thread_info_t>(&basic),
                                    &basic_count) != KERN_SUCCESS) {
                        mach_port_deallocate(mach_task_self(), thread);
                        continue;
                    }

                    char name_buffer[128]{};
                    if (pthread_t pthread = pthread_from_mach_thread_np(thread);
                        pthread != nullptr) {
                        pthread_getname_np(pthread, name_buffer, sizeof(name_buffer));
                    }
                    std::string name = name_buffer[0] != '\0'
                                           ? std::string{name_buffer}
                                           : "thread-" + std::to_string(identifier.thread_id);
                    thread_samples.insert_or_assign(
                        identifier.thread_id,
                        ThreadSample{.name = std::move(name),
                                     .cpu_percent = 100.0f * static_cast<float>(basic.cpu_usage) /
                                                    static_cast<float>(TH_USAGE_SCALE),
                                     .run_state = basic.run_state});

                    if (basic.run_state == TH_STATE_RUNNING) {
#if defined(__x86_64__)
                        x86_thread_state64_t state{};
                        mach_msg_type_number_t state_count = x86_THREAD_STATE64_COUNT;
                        if (thread_get_state(thread, x86_THREAD_STATE64,
                                             reinterpret_cast<thread_state_t>(&state),
                                             &state_count) == KERN_SUCCESS) {
                            // Group nearby instructions into one bucket so a hot function is not
                            // split into dozens of almost-identical rows.
                            const u64 pc = static_cast<u64>(state.__rip) & ~0x3fULL;
                            ++sample_counts[{identifier.thread_id, pc}];
                        }
#endif
                    }
                    mach_port_deallocate(mach_task_self(), thread);
                }
                vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(task_thread_list),
                              static_cast<vm_size_t>(thread_count) * sizeof(thread_t));
            }

            AddTime(
                TimeMetric::SamplerOverhead,
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - sample_begin));
            const auto now = Clock::now();
            if (now >= next_flush) {
                FlushThreadSamples(now, sample_counts, thread_samples);
                sample_counts.clear();
                thread_samples.clear();
                next_flush = now + std::chrono::seconds(1);
            }
            std::this_thread::sleep_until(next_sample);
        }

        if (!sample_counts.empty() || !thread_samples.empty()) {
            FlushThreadSamples(Clock::now(), sample_counts, thread_samples);
        }
        mach_port_deallocate(mach_task_self(), own_thread);
    }
#endif

private:
    std::atomic_bool enabled{};
    std::atomic<u64> recorded_frames{};
    std::array<std::atomic<u64>, CounterCount> counters{};
    std::array<std::atomic<s64>, TimeMetricCount> timings{};
    std::mutex lifecycle_mutex;
    GpuWaitLog gpu_wait_log;
    std::ofstream frames;
    std::ofstream samples;
    std::ofstream threads;
    std::ofstream metadata;
    std::filesystem::path frame_path;
    std::filesystem::path sample_path;
    std::filesystem::path thread_path;
    std::filesystem::path meta_path;
    Clock::time_point start_time{};
    Clock::time_point last_frame_time{};
    Clock::time_point last_frame_flush{};
    ProcessSnapshot last_process{};
    Common::GuestTimeStallTracker::Duration last_guest_stall{};
    u64 frame_number{};
#ifdef __APPLE__
    std::jthread sampler;
#endif
};

Recorder& GetRecorder() {
    static Recorder recorder;
    return recorder;
}

} // namespace

void Start() {
    GetRecorder().Start();
}

void Stop() {
    GetRecorder().Stop();
}

void SetStartRequested(bool requested) noexcept {
    start_requested.store(requested, std::memory_order_release);
}

void SetOutputDirectory(std::string directory) {
    output_directory = std::move(directory);
}

bool IsStartRequested() noexcept {
    return start_requested.load(std::memory_order_acquire);
}

bool IsEnabled() noexcept {
    return GetRecorder().IsEnabled();
}

u64 GetRecordedFrameCount() noexcept {
    return GetRecorder().GetFrameCount();
}

void Increment(Counter counter, u64 amount) noexcept {
    GetRecorder().Increment(counter, amount);
}

void AddTime(TimeMetric metric, std::chrono::nanoseconds duration) noexcept {
    GetRecorder().AddTime(metric, duration);
}

void RecordGpuWait(const GpuWaitEvent& event) noexcept {
    GetRecorder().RecordGpuWait(event);
}

void RecordFrame(u32 pending_flips, u32 request_depth, u32 game_width, u32 game_height,
                 u32 output_width, u32 output_height, const FlipTiming& flip_timing) {
    GetRecorder().RecordFrame(pending_flips, request_depth, game_width, game_height, output_width,
                              output_height, flip_timing);
}

} // namespace Core::PerfTelemetry
