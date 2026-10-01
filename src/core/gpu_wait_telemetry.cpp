// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/gpu_wait_log.h"
#include "core/performance_telemetry.h"

#include <functional>
#include <iomanip>
#include <sstream>
#include <string_view>

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace Core::PerfTelemetry {
namespace {
constexpr std::string_view Header =
    "event,start_ms,end_ms,wait_ms,thread_id,timeline_id,target_tick,known_gpu_tick,"
    "current_tick,source,resource,resource_id,capacity_bytes,request_bytes,offset_bytes,"
    "context_bits,image_address,image_bytes,caller_file,caller_line,caller_function,"
    "texture_lock_file,texture_lock_line,texture_lock_function\n";

std::string_view SourceName(GpuWaitSource source) {
    switch (source) {
    case GpuWaitSource::StreamReuse:
        return "stream_reuse";
    case GpuWaitSource::SchedulerFinish:
        return "scheduler_finish";
    case GpuWaitSource::CommandPool:
        return "command_pool";
    case GpuWaitSource::DescriptorRetirement:
        return "descriptor_retirement";
    case GpuWaitSource::FaultBufferReuse:
        return "fault_buffer_reuse";
    case GpuWaitSource::PriorityCallback:
        return "priority_callback";
    default:
        return "unspecified";
    }
}

std::string_view ResourceName(GpuWaitResource resource) {
    switch (resource) {
    case GpuWaitResource::UploadStaging:
        return "upload_staging";
    case GpuWaitResource::UniformStream:
        return "uniform_stream";
    case GpuWaitResource::QuadIndex:
        return "quad_index";
    case GpuWaitResource::Download:
        return "download";
    case GpuWaitResource::DeviceUtility:
        return "device_utility";
    case GpuWaitResource::TileScratch:
        return "tile_scratch";
    default:
        return "unspecified";
    }
}

// CSV doubles embedded quotes (std::quoted would use backslash escapes).
void CsvString(std::ostream& out, std::string_view value) {
    out << '"';
    for (const char c : value) {
        if (c == '"') {
            out << '"';
        }
        out << c;
    }
    out << '"';
}

void Location(std::ostream& out, const std::source_location& location) {
    CsvString(out, location.file_name());
    out << ',' << location.line() << ',';
    CsvString(out, location.function_name());
}
} // namespace

ScopedGpuWait::ScopedGpuWait(u64 timeline_id, u64 target_tick, u64 known_gpu_tick, u64 current_tick,
                             const GpuWaitInfo& info, std::source_location caller) noexcept
    : active{IsEnabled()} {
    if (!active) {
        return;
    }
    event.timeline_id = timeline_id;
    event.target_tick = target_tick;
    event.known_gpu_tick = known_gpu_tick;
    event.current_tick = current_tick;
    event.info = info;
    event.caller = caller;
#ifdef __APPLE__
    pthread_threadid_np(nullptr, &event.thread_id);
#else
    event.thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    ScopedGpuWaitContext::Capture(event);
    Increment(Counter::GpuWaits);
    event.begin = GpuWaitEvent::Clock::now();
}

ScopedGpuWait::~ScopedGpuWait() {
    if (active) {
        event.end = GpuWaitEvent::Clock::now();
        AddTime(TimeMetric::GpuWait,
                std::chrono::duration_cast<std::chrono::nanoseconds>(event.end - event.begin));
        RecordGpuWait(event);
    }
}

GpuWaitLog::GpuWaitLog() {
    queued.reserve(QueueCapacity);
    draining.reserve(QueueCapacity);
}

GpuWaitLog::~GpuWaitLog() {
    Close();
}

bool GpuWaitLog::Open(const std::filesystem::path& csv, GpuWaitEvent::Clock::time_point start,
                      size_t limit) {
    Close();
    if (limit < Header.size() + 4096 || limit > FileLimit) {
        return false;
    }
    csv_path = csv;
    previous_path = csv.parent_path() / (csv.stem().string() + ".previous.csv");
    status_path = csv.parent_path() / (csv.stem().string() + "_status.txt");
    start_time = start;
    file_limit = limit;
    written_bytes = Header.size();
    written_events = current_events = previous_events = evicted_events = rotations = 0;
    dropped_busy = dropped_full = dropped_io = 0;
    io_failed = false;
    stream.clear();
    stream.open(csv_path, std::ios::out | std::ios::trunc | std::ios::binary);
    stream << Header;
    stream.flush();
    if (!stream) {
        stream.close();
        return false;
    }
    // Every session uses a new stem; also make an explicit reopen start with a clean history.
    std::error_code error;
    std::filesystem::remove(previous_path, error);
    if (error) {
        stream.close();
        return false;
    }
    {
        std::scoped_lock lock{queue_mutex};
        queued.clear();
        accepting = true;
    }
    WriteStatus();
    writer = std::jthread([this](std::stop_token token) { Run(token); });
    return true;
}

void GpuWaitLog::Close() {
    {
        std::scoped_lock lock{queue_mutex};
        accepting = false;
    }
    if (writer.joinable()) {
        writer.request_stop();
        writer.join();
    }
    if (stream.is_open()) {
        stream.close();
    }
}

void GpuWaitLog::Record(const GpuWaitEvent& event) noexcept {
    if (!accepting.load(std::memory_order_relaxed)) {
        return;
    }
    if (io_failed.load(std::memory_order_relaxed)) {
        dropped_io.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // No producer waits on the journal mutex, allocates storage, or performs file I/O.
    std::unique_lock lock{queue_mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
        dropped_busy.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (!accepting || event.begin < start_time) {
        return;
    }
    if (queued.size() == QueueCapacity) {
        dropped_full.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    queued.push_back(event);
}

bool GpuWaitLog::Rotate() {
    stream.close();
    std::error_code error;
    std::filesystem::remove(previous_path, error);
    if (error) {
        return false;
    }
    std::filesystem::rename(csv_path, previous_path, error);
    if (error) {
        return false;
    }
    stream.clear();
    stream.open(csv_path, std::ios::out | std::ios::trunc | std::ios::binary);
    stream << Header;
    evicted_events += previous_events;
    previous_events = current_events;
    current_events = 0;
    written_bytes = Header.size();
    ++rotations;
    return bool(stream);
}

void GpuWaitLog::Drain() {
    {
        std::scoped_lock lock{queue_mutex};
        queued.swap(draining); // Both vectors retain their preallocated bounded capacity.
    }
    for (const auto& event : draining) {
        if (io_failed) {
            ++dropped_io;
            continue;
        }
        const auto ms = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        std::ostringstream row;
        row << std::fixed << std::setprecision(6) << written_events + 1 << ','
            << ms(event.begin - start_time) << ',' << ms(event.end - start_time) << ','
            << ms(event.end - event.begin) << ',' << event.thread_id << ',' << event.timeline_id
            << ',' << event.target_tick << ',' << event.known_gpu_tick << ',' << event.current_tick
            << ',' << SourceName(event.info.source) << ',' << ResourceName(event.info.resource)
            << ',' << event.info.resource_id << ',' << event.info.capacity_bytes << ','
            << event.info.request_bytes << ',' << event.info.offset_bytes << ','
            << event.context_bits << ',' << event.image_address << ',' << event.image_bytes << ',';
        Location(row, event.caller);
        row << ',';
        Location(row, event.texture_lock_site);
        row << '\n';
        const auto line = row.str();
        if (line.size() + Header.size() > file_limit ||
            (line.size() + written_bytes > file_limit && !Rotate())) {
            io_failed = true;
            ++dropped_io;
            continue;
        }
        stream << line;
        if (!stream) {
            io_failed = true;
            ++dropped_io;
            continue;
        }
        written_bytes += line.size();
        ++written_events;
        ++current_events;
    }
    draining.clear();
    stream.flush();
    if (!stream) {
        io_failed = true;
    }
}

void GpuWaitLog::WriteStatus() {
    std::ofstream status{status_path, std::ios::out | std::ios::trunc};
    status << "version=1\nfile_limit_bytes=" << file_limit << "\nmax_csv_files=2\n"
           << "queue_capacity=" << QueueCapacity << "\nwritten_events=" << written_events
           << "\ncurrent_events=" << current_events << "\nprevious_events=" << previous_events
           << "\nevicted_events=" << evicted_events << "\nrotations=" << rotations
           << "\ndropped_busy=" << dropped_busy.load() << "\ndropped_full=" << dropped_full.load()
           << "\ndropped_io=" << dropped_io.load() << "\nio_failed=" << io_failed.load()
           << "\naccepting=" << accepting.load() << '\n';
}

void GpuWaitLog::Run(std::stop_token token) {
#ifdef __APPLE__
    pthread_setname_np("shadPS4:GpuWaitWriter");
#endif
    auto next_status = GpuWaitEvent::Clock::now() + std::chrono::seconds(1);
    while (!token.stop_requested()) {
        std::unique_lock lock{wake_mutex};
        wake_cv.wait_for(lock, token, std::chrono::milliseconds(100), [] { return false; });
        lock.unlock();
        Drain();
        const auto now = GpuWaitEvent::Clock::now();
        if (now >= next_status) {
            WriteStatus();
            next_status = now + std::chrono::seconds(1);
        }
    }
    Drain();
    WriteStatus();
}

} // namespace Core::PerfTelemetry
