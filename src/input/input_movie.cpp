// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "input/input_movie.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

namespace Input::Movie {
namespace {
static_assert(std::endian::native == std::endian::little);
constexpr u64 HashSeed = 14695981039346656037ULL;
u64 Hash(u64 value, const void* data, size_t size) {
    for (auto byte : std::span{static_cast<const u8*>(data), size}) {
        value = (value ^ byte) * 1099511628211ULL;
    }
    return value;
}
u64 SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
Pad Pack(const Libraries::Pad::OrbisPadData& data) {
    Pad p{};
    p.buttons = u32(data.buttons);
    p.axes = {data.leftStick.x,  data.leftStick.y,      data.rightStick.x,
              data.rightStick.y, data.analogButtons.l2, data.analogButtons.r2};
    p.connected = data.connected;
    p.connected_count = data.connectedCount;
    p.orientation = {data.orientation.x, data.orientation.y, data.orientation.z,
                     data.orientation.w};
    p.acceleration = {data.acceleration.x, data.acceleration.y, data.acceleration.z};
    p.angular_velocity = {data.angularVelocity.x, data.angularVelocity.y, data.angularVelocity.z};
    p.touch_count = data.touchData.touchNum;
    p.touch_held = data.touchData.time_since_touch_held_down;
    for (u32 i = 0; i < 2; ++i) {
        p.touches[i].x = data.touchData.touch[i].x;
        p.touches[i].y = data.touchData.touch[i].y;
        p.touches[i].id = data.touchData.touch[i].id;
    }
    p.timestamp = data.timestamp;
    return p;
}
Libraries::Pad::OrbisPadData Unpack(const Pad& p, u64 timestamp) {
    Libraries::Pad::OrbisPadData data{};
    data.buttons = Libraries::Pad::OrbisPadButtonDataOffset(p.buttons);
    data.leftStick = {p.axes[0], p.axes[1]};
    data.rightStick = {p.axes[2], p.axes[3]};
    data.analogButtons.l2 = p.axes[4];
    data.analogButtons.r2 = p.axes[5];
    data.orientation = {p.orientation[0], p.orientation[1], p.orientation[2], p.orientation[3]};
    data.acceleration = {p.acceleration[0], p.acceleration[1], p.acceleration[2]};
    data.angularVelocity = {p.angular_velocity[0], p.angular_velocity[1], p.angular_velocity[2]};
    data.touchData.touchNum = p.touch_count;
    data.touchData.time_since_touch_held_down = p.touch_held;
    for (u32 i = 0; i < 2; ++i) {
        data.touchData.touch[i].x = p.touches[i].x;
        data.touchData.touch[i].y = p.touches[i].y;
        data.touchData.touch[i].id = p.touches[i].id;
    }
    data.connected = p.connected;
    data.connectedCount = p.connected_count;
    data.timestamp = timestamp;
    return data;
}
Session* session{};
} // namespace

struct Session::Impl {
    bool replay{};
    std::atomic_bool stopping{}, complete{}, aborted{};
    std::atomic<u32> failure{}; // 1: I/O, 2: byte cap, 3: queue overflow
    std::atomic<u64> frame{}, written{}, state_count{}, poll_count{};
    Header header;
    std::filesystem::path directory;
    std::ofstream stream;
    size_t file_limit{FileLimit};
    u64 file_bytes{sizeof(Header)}, hash{HashSeed}, end_frame{}, end_us{};
    std::mutex mutex;
    std::condition_variable_any wake;
    std::vector<Event> queued, draining;
    std::array<std::optional<Pad>, 5> previous;
    std::array<u64, 5> previous_frame{};
    std::array<std::vector<Event>, 5> states;
    std::array<size_t, 5> cursor{}, latest_cursor{};
    std::array<u64, 5> last_timestamp{};
    std::array<bool, 5> neutral_sent{};
    std::function<void()> quit;
    std::function<void()> screenshot;
    u32 screenshot_count{};
    std::jthread worker;

    u64 Elapsed() const {
        return (SteadyNs() - header.start_steady_ns) / 1000;
    }
    void Status() {
        const auto temporary = directory / "status.json.tmp";
        std::ofstream out{temporary};
        out << "{\"version\":1,\"mode\":\"" << (replay ? "replay" : "record")
            << "\",\"valid\":" << (failure == 0 ? "true" : "false")
            << ",\"complete\":" << (complete ? "true" : "false") << ",\"error_code\":" << failure
            << ",\"frame\":" << frame << ",\"polls\":" << poll_count
            << ",\"states\":" << state_count << ",\"events_written\":" << written
            << ",\"bytes\":" << file_bytes << ",\"file_limit\":" << file_limit
            << ",\"end_frame\":" << end_frame
            << ",\"replay_finished\":" << (replay && frame >= end_frame ? "true" : "false")
            << ",\"aborted\":" << (aborted ? "true" : "false") << "}\n";
        out.close();
        std::error_code error;
        if (out)
            std::filesystem::rename(temporary, directory / "status.json", error);
        if (!out || error)
            failure = 1;
    }
    // Producer holds only the short queue mutex. File I/O/hashing never holds it.
    void Enqueue(Event event) {
        if (failure || stopping)
            return;
        if (queued.size() == QueueCapacity) {
            failure = 3; // A lossy movie must never be offered as replayable.
            return;
        }
        queued.push_back(event);
    }
    void Write(const Event& event, bool footer = false) {
        if (failure)
            return;
        if (file_bytes + sizeof(Event) + (footer ? 0 : sizeof(Event)) > file_limit) {
            failure = 2;
            return;
        }
        stream.write(reinterpret_cast<const char*>(&event), sizeof(event));
        if (!stream) {
            failure = 1;
            return;
        }
        if (!footer)
            hash = Hash(hash, &event, sizeof(event));
        file_bytes += sizeof(event);
        ++written;
        if (event.type == EventType::State)
            ++state_count;
    }
    void Run(std::stop_token stop) {
        while (!stop.stop_requested()) {
            {
                std::unique_lock lock{mutex};
                wake.wait_for(lock, stop, std::chrono::milliseconds{100}, [] { return false; });
                queued.swap(draining);
            }
            for (const auto& event : draining)
                Write(event);
            draining.clear();
            if (!replay)
                stream.flush();
            if (!replay && !stream)
                failure = 1;
            {
                std::error_code error;
                if (screenshot_count < 8 &&
                    std::filesystem::remove(directory / "screenshot", error) && !error) {
                    ++screenshot_count;
                    if (screenshot)
                        screenshot();
                }
            }
            if (replay && !aborted) {
                std::error_code error;
                if (std::filesystem::exists(directory / "stop", error) && !error) {
                    aborted = true;
                    if (quit)
                        quit();
                }
            }
            Status();
        }
    }
};

Session::Session() : impl{std::make_unique<Impl>()} {}
Session::~Session() {
    Stop();
}
bool Session::OpenRecord(const std::filesystem::path& directory, std::string& error, size_t limit) {
    auto& p = *impl;
    if (p.worker.joinable()) {
        error = "Input movie already open";
        return false;
    }
    p.directory = directory;
    p.file_limit = std::clamp(limit, sizeof(Header) + 2 * sizeof(Event), FileLimit);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec || std::filesystem::exists(directory / "controller.bin")) {
        error = "Cannot create a new input recording in " + directory.string();
        return false;
    }
    p.header.start_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
    p.header.start_steady_ns = SteadyNs();
    p.stream.open(directory / "controller.bin", std::ios::binary | std::ios::trunc);
    p.stream.write(reinterpret_cast<const char*>(&p.header), sizeof(Header));
    p.hash = Hash(HashSeed, &p.header, sizeof(Header));
    p.stream.flush();
    if (!p.stream) {
        error = "Cannot write input movie header";
        return false;
    }
    p.queued.reserve(QueueCapacity);
    p.draining.reserve(QueueCapacity);
    p.Status();
    p.worker = std::jthread([&p](std::stop_token token) { p.Run(token); });
    return true;
}

bool Session::OpenReplay(const std::filesystem::path& file, const std::filesystem::path& report,
                         std::function<void()> quit, std::string& error) {
    auto& p = *impl;
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    if (ec || size < sizeof(Header) + sizeof(Event) || size > FileLimit ||
        (size - sizeof(Header)) % sizeof(Event)) {
        error = "Truncated or oversized input movie";
        return false;
    }
    std::ifstream in{file, std::ios::binary};
    in.read(reinterpret_cast<char*>(&p.header), sizeof(Header));
    const Header expected;
    if (!in || p.header.magic != expected.magic || p.header.version != 1 ||
        p.header.header_size != sizeof(Header) || p.header.event_size != sizeof(Event) ||
        p.header.slots != 5) {
        error = "Unsupported input movie format";
        return false;
    }
    u64 checksum = Hash(HashSeed, &p.header, sizeof(Header));
    u64 frame{}, elapsed{};
    bool ended = false;
    for (u64 i = sizeof(Header); i < size; i += sizeof(Event)) {
        Event event;
        in.read(reinterpret_cast<char*>(&event), sizeof(event));
        if (!in || ended || event.frame < frame || event.elapsed_us < elapsed) {
            error = "Invalid input movie ordering";
            return false;
        }
        if (event.type == EventType::End) {
            if (event.pad.timestamp != checksum || i + sizeof(Event) != size ||
                event.frame != frame) {
                error = "Input movie checksum/footer mismatch";
                return false;
            }
            ended = true;
            p.end_frame = event.frame;
            p.end_us = event.elapsed_us;
        } else if (event.type == EventType::Flip) {
            if (event.frame != frame + 1) {
                error = "Input movie skipped a guest flip";
                return false;
            }
        } else if (event.type == EventType::State && event.slot < 5 && event.frame == frame &&
                   event.pad.connected <= 1 && event.pad.touch_count <= 2) {
            p.states[event.slot].push_back(event);
            ++p.state_count;
        } else {
            error = "Invalid input movie event";
            return false;
        }
        if (!ended)
            checksum = Hash(checksum, &event, sizeof(event));
        frame = event.frame;
        elapsed = event.elapsed_us;
    }
    if (!ended || !p.state_count) {
        error = "Incomplete/empty input movie";
        return false;
    }
    p.replay = true;
    p.directory = report;
    p.quit = std::move(quit);
    std::filesystem::create_directories(report, ec);
    if (ec) {
        error = "Cannot create replay status directory";
        return false;
    }
    p.file_bytes = size;
    p.queued.reserve(QueueCapacity);
    p.draining.reserve(QueueCapacity);
    p.Status();
    p.worker = std::jthread([&p](std::stop_token token) { p.Run(token); });
    return true;
}

void Session::Stop() {
    auto& p = *impl;
    if (!p.worker.joinable())
        return;
    p.stopping = true;
    p.worker.request_stop();
    p.wake.notify_one();
    p.worker.join();
    if (!p.replay) {
        std::scoped_lock lock{p.mutex};
        for (const auto& event : p.queued)
            p.Write(event);
        p.queued.clear();
        Event footer{.type = EventType::End, .frame = p.frame, .elapsed_us = p.Elapsed()};
        footer.pad.timestamp = p.hash;
        p.Write(footer, true);
        p.stream.flush();
        if (!p.stream)
            p.failure = 1;
        p.stream.close();
    }
    p.complete = true;
    p.Status();
}

void Session::Flip() {
    auto& p = *impl;
    if (p.stopping)
        return;
    if (p.replay) {
        ++p.frame;
        return;
    }
    std::scoped_lock lock{p.mutex};
    if (p.stopping)
        return;
    ++p.frame;
    p.Enqueue({.type = EventType::Flip, .frame = p.frame, .elapsed_us = p.Elapsed()});
}

void Session::Record(u32 slot, std::span<const Libraries::Pad::OrbisPadData> data) {
    auto& p = *impl;
    if (p.replay || p.stopping || p.failure || slot >= 5)
        return;
    ++p.poll_count;
    std::scoped_lock lock{p.mutex};
    for (const auto& value : data) {
        Pad packed = Pack(value);
        Pad comparable = packed;
        comparable.timestamp = 0;
        if (p.previous[slot] && p.previous_frame[slot] == p.frame &&
            std::memcmp(&*p.previous[slot], &comparable, sizeof(Pad)) == 0)
            continue;
        p.previous[slot] = comparable;
        p.previous_frame[slot] = p.frame;
        p.Enqueue({.type = EventType::State,
                   .slot = slot,
                   .frame = p.frame,
                   .elapsed_us = p.Elapsed(),
                   .pad = packed});
    }
}

bool Session::Connection(u32 slot, bool& connected, u8& count) {
    auto& p = *impl;
    if (!p.replay || slot >= 5)
        return false;
    std::scoped_lock lock{p.mutex};
    const auto& events = p.states[slot];
    const auto next =
        std::upper_bound(events.begin(), events.end(), p.frame.load(),
                         [](u64 frame, const Event& event) { return frame < event.frame; });
    const auto& pad =
        events.empty() ? Pad{} : (next == events.begin() ? events.front() : *std::prev(next)).pad;
    connected = pad.connected;
    count = pad.connected_count;
    return true;
}

std::optional<int> Session::Read(u32 slot, Libraries::Pad::OrbisPadData* data, int capacity,
                                 bool latest, u64 timestamp) {
    auto& p = *impl;
    if (!p.replay)
        return std::nullopt;
    if (slot >= 5 || !data || capacity < 1 || capacity > 64)
        return 0;
    ++p.poll_count;
    std::scoped_lock lock{p.mutex};
    const u64 frame = p.frame;
    auto& events = p.states[slot];
    const auto emit = [&](const Pad& pad, int i) {
        p.last_timestamp[slot] = std::max(timestamp, p.last_timestamp[slot] + 1);
        data[i] = Unpack(pad, p.last_timestamp[slot]);
    };
    if (p.aborted || p.stopping || frame >= p.end_frame) {
        if (!latest && p.neutral_sent[slot])
            return 0;
        Pad neutral;
        if (!events.empty()) {
            neutral.connected = events.back().pad.connected;
            neutral.connected_count = events.back().pad.connected_count;
        }
        emit(neutral, 0);
        p.neutral_sent[slot] = true;
        return 1;
    }
    if (latest) {
        auto& current = p.latest_cursor[slot];
        while (current < events.size() && events[current].frame < frame)
            ++current;
        // Preserve press/release edges observed by repeated ReadState calls in
        // one guest frame. Extra polls hold the last value instead of consuming
        // input belonging to future frames.
        if (current < events.size() && events[current].frame == frame)
            ++current;
        Pad neutral;
        if (!events.empty()) {
            neutral.connected = events.front().pad.connected;
            neutral.connected_count = events.front().pad.connected_count;
        }
        emit(current ? events[current - 1].pad : neutral, 0);
        return 1;
    }
    auto& current = p.cursor[slot];
    int count{};
    while (current < events.size() && events[current].frame <= frame && count < capacity) {
        emit(events[current++].pad, count++);
    }
    return count;
}

bool Session::Replaying() const {
    return impl->replay;
}
u64 Session::Frame() const {
    return impl->frame;
}
void Session::SetScreenshotCallback(std::function<void()> callback) {
    // Configure before requests can be placed by the collector.
    impl->screenshot = std::move(callback);
}

bool Configure(const std::filesystem::path& record, const std::filesystem::path& replay,
               const std::filesystem::path& report, std::function<void()> quit,
               std::function<void()> screenshot, std::string& error) {
    if (record.empty() && replay.empty())
        return true;
    if (session || (!record.empty() && !replay.empty()) || (!replay.empty() && report.empty())) {
        error = "Input recording and replay require separate, exclusive sessions";
        return false;
    }
    auto next = std::make_unique<Session>();
    next->SetScreenshotCallback(std::move(screenshot));
    const bool ok = replay.empty() ? next->OpenRecord(record, error)
                                   : next->OpenReplay(replay, report, std::move(quit), error);
    if (!ok)
        return false;
    session = next.release();
    std::atexit(Stop);
    return true;
}
bool Active() noexcept {
    return session != nullptr;
}
bool Replaying() noexcept {
    return session && session->Replaying();
}
void Stop() {
    if (session)
        session->Stop();
}
void Flip() {
    if (session)
        session->Flip();
}
void Record(u32 slot, std::span<const Libraries::Pad::OrbisPadData> data) {
    if (session)
        session->Record(slot, data);
}
std::optional<int> Read(u32 slot, Libraries::Pad::OrbisPadData* data, int capacity, bool latest,
                        u64 timestamp) {
    return session ? session->Read(slot, data, capacity, latest, timestamp) : std::nullopt;
}
bool Connection(u32 slot, bool& connected, u8& count) {
    return session && session->Connection(slot, connected, count);
}
} // namespace Input::Movie
