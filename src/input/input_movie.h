// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include "core/libraries/pad/pad.h"

namespace Input::Movie {
// Explicit little-endian wire layout. Do not serialize OrbisPadData padding or
// guest-owned reserved/device-extension bytes which ProcessStates does not set.
struct Pad {
    u32 buttons{};
    std::array<u8, 6> axes{128, 128, 128, 128, 0, 0};
    u8 connected{}, connected_count{};
    std::array<float, 4> orientation{0, 0, 0, 1};
    std::array<float, 3> acceleration{}, angular_velocity{};
    u32 touch_held{};
    u8 touch_count{}, reserved[3]{};
    struct Touch {
        u16 x{}, y{};
        u8 id{}, reserved[3]{};
    } touches[2];
    u32 reserved2{};
    u64 timestamp{};
};
static_assert(sizeof(Pad) == 88);
struct Header {
    std::array<char, 8> magic{'S', 'H', 'A', 'D', 'P', 'A', 'D', '1'};
    u32 version{1}, header_size{64}, event_size{112}, slots{5};
    u64 start_unix_ns{}, start_steady_ns{}, reserved[3]{};
};
static_assert(sizeof(Header) == 64);
enum class EventType : u32 { State = 1, Flip = 2, End = 3 };
struct Event {
    EventType type{};
    u32 slot{};
    u64 frame{}, elapsed_us{};
    Pad pad{};
};
static_assert(sizeof(Event) == 112);

class Session {
public:
    static constexpr size_t FileLimit = 64 * 1024 * 1024;
    static constexpr size_t QueueCapacity = 4096;
    Session();
    ~Session();
    bool OpenRecord(const std::filesystem::path& directory, std::string& error,
                    size_t file_limit = FileLimit);
    bool OpenReplay(const std::filesystem::path& file, const std::filesystem::path& report,
                    std::function<void()> quit, std::string& error);
    void SetScreenshotCallback(std::function<void()> callback);
    void Stop();
    void Flip();
    void Record(u32 slot, std::span<const Libraries::Pad::OrbisPadData> data);
    std::optional<int> Read(u32 slot, Libraries::Pad::OrbisPadData* data, int capacity, bool latest,
                            u64 timestamp);
    bool Connection(u32 slot, bool& connected, u8& count);
    bool Replaying() const;
    u64 Frame() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Configured before guest execution. Session storage lives until process exit so
// guest threads cannot access freed replay data during quick-exit shutdown.
bool Configure(const std::filesystem::path& record, const std::filesystem::path& replay,
               const std::filesystem::path& report, std::function<void()> quit,
               std::function<void()> screenshot, std::string& error);
bool Active() noexcept;
bool Replaying() noexcept;
void Stop();
void Flip();
void Record(u32 slot, std::span<const Libraries::Pad::OrbisPadData> data);
std::optional<int> Read(u32 slot, Libraries::Pad::OrbisPadData* data, int capacity, bool latest,
                        u64 timestamp);
bool Connection(u32 slot, bool& connected, u8& count);
} // namespace Input::Movie
