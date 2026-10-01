// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "lbp3_online_bridge.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <fcntl.h>
#endif

#include "common/arch.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "core/lbp3_online.h"
#include "sockets.h"

#if __has_include(<httplib.h>)
#include <httplib.h>
#define LBP3_BRIDGE_WITH_HTTPLIB 1
#endif

namespace Libraries::Net::Lbp3OnlineBridge {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<u8, 4> FrameMagic{'L', 'B', '3', 'B'};
constexpr u8 FrameVersion = 1;
constexpr u8 FrameTypeDatagram = 1;
constexpr u8 FrameTypeHello = 2;
constexpr u8 FrameTypeHelloAck = 3;
constexpr size_t FrameHeaderSize = 28;
constexpr size_t MaxPayloadSize = 4096;
constexpr size_t MaxQueuedDatagrams = 256;
constexpr u16 DefaultGamePort = 3658;
constexpr u16 DefaultHelperPort = 46973;
constexpr auto HelloInterval = std::chrono::seconds(2);
constexpr auto HelloAckTimeout = std::chrono::seconds(6);
constexpr auto ServiceInterval = std::chrono::milliseconds(25);

// Guest virtual addresses for CUSA00063 01.26; they exclude the SELF file bias (0x4000).
constexpr uintptr_t MatchingDispatcherOffset = 0x338cc0;
constexpr uintptr_t FindBestRoomBuilderOffset = 0x33a830;
constexpr uintptr_t VectorPushU64Offset = 0xb04e70;
constexpr uintptr_t OnlineManagerOffset = 0x130d6b8;
constexpr uintptr_t MatchingObjectPointerOffset = 0x20;
constexpr uintptr_t MatchingQueueOffset = 0x230;
constexpr uintptr_t MatchingQueueSizeOffset = 0x238;
constexpr uintptr_t MatchingQueueCapacityOffset = 0x23c;
constexpr uintptr_t MatchingActiveStateOffset = 0x30;
constexpr uintptr_t MatchingBlockingStateOffset = 0x110;
constexpr u64 FindBestRoomSelector = 4;

#ifdef _WIN32
constexpr net_socket InvalidSocket = INVALID_SOCKET;
#else
constexpr net_socket InvalidSocket = -1;
#endif

struct Frame {
    u8 type{};
    Channel channel{};
    u8 flags{};
    Endpoint source{};
    Endpoint destination{};
    std::vector<u8> payload;
};

struct BridgeState {
    std::mutex mutex;
    std::mutex find_best_room_mutex;
    std::jthread service_thread;
    net_socket socket{InvalidSocket};
    sockaddr_in helper{};
    bool endpoint_configured{};
    bool connected{};
    u32 advertised_addr{};
    std::string online_id;
    Clock::time_point last_hello{};
    Clock::time_point last_hello_ack{};
    std::array<std::deque<Frame>, 3> queues;
    std::atomic_bool update_my_player_data_seen{false};
    std::atomic_bool find_best_room_queued{false};
    std::atomic_bool find_best_room_signatures_rejected{false};
    Clock::time_point helper_roster_last_check{};
    Clock::time_point helper_roster_first_seen{};
    bool helper_roster_requests_join{};
    std::string helper_roster_local_online_id;

    ~BridgeState() {
        if (service_thread.joinable()) {
            service_thread.request_stop();
            service_thread.join();
        }
#ifdef _WIN32
        if (socket != InvalidSocket) {
            closesocket(socket);
        }
#else
        if (socket != InvalidSocket) {
            close(socket);
        }
#endif
    }
};

void ServiceLoop(BridgeState& state, std::stop_token stop_token);

BridgeState& State() {
    static BridgeState state;
    return state;
}

void WriteU16(std::vector<u8>& out, size_t offset, u16 value) {
    out[offset] = static_cast<u8>(value >> 8);
    out[offset + 1] = static_cast<u8>(value);
}

void WriteU32(std::vector<u8>& out, size_t offset, u32 value) {
    out[offset] = static_cast<u8>(value >> 24);
    out[offset + 1] = static_cast<u8>(value >> 16);
    out[offset + 2] = static_cast<u8>(value >> 8);
    out[offset + 3] = static_cast<u8>(value);
}

u16 ReadU16(const u8* data, size_t offset) {
    return static_cast<u16>((static_cast<u16>(data[offset]) << 8) | data[offset + 1]);
}

u32 ReadU32(const u8* data, size_t offset) {
    return (static_cast<u32>(data[offset]) << 24) | (static_cast<u32>(data[offset + 1]) << 16) |
           (static_cast<u32>(data[offset + 2]) << 8) | data[offset + 3];
}

bool IsValidOnlineId(std::string_view value) {
    if (value.empty() || value.size() > 16) {
        return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = value[i];
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        if (!alpha && !digit && !(i != 0 && (c == '-' || c == '_'))) {
            return false;
        }
    }
    return true;
}

std::vector<u8> EncodeFrame(const Frame& frame) {
    if (frame.payload.size() > MaxPayloadSize) {
        return {};
    }
    std::vector<u8> out(FrameHeaderSize + frame.payload.size());
    std::copy(FrameMagic.begin(), FrameMagic.end(), out.begin());
    out[4] = FrameVersion;
    out[5] = frame.type;
    out[6] = static_cast<u8>(frame.channel);
    out[7] = frame.flags;
    WriteU32(out, 8, ntohl(frame.source.addr));
    WriteU32(out, 12, ntohl(frame.destination.addr));
    WriteU16(out, 16, ntohs(frame.source.port));
    WriteU16(out, 18, ntohs(frame.destination.port));
    WriteU16(out, 20, ntohs(frame.source.vport));
    WriteU16(out, 22, ntohs(frame.destination.vport));
    WriteU16(out, 24, static_cast<u16>(frame.payload.size()));
    std::copy(frame.payload.begin(), frame.payload.end(), out.begin() + FrameHeaderSize);
    return out;
}

bool DecodeFrame(const u8* data, size_t size, Frame* out) {
    if (data == nullptr || out == nullptr || size < FrameHeaderSize ||
        !std::equal(FrameMagic.begin(), FrameMagic.end(), data) || data[4] != FrameVersion) {
        return false;
    }
    const size_t payload_size = ReadU16(data, 24);
    if (payload_size > MaxPayloadSize || size != FrameHeaderSize + payload_size ||
        data[5] < FrameTypeDatagram || data[5] > FrameTypeHelloAck ||
        data[6] > static_cast<u8>(Channel::Control) || data[26] != 0 || data[27] != 0) {
        return false;
    }
    out->type = data[5];
    out->channel = static_cast<Channel>(data[6]);
    out->flags = data[7];
    out->source.addr = htonl(ReadU32(data, 8));
    out->destination.addr = htonl(ReadU32(data, 12));
    out->source.port = htons(ReadU16(data, 16));
    out->destination.port = htons(ReadU16(data, 18));
    out->source.vport = htons(ReadU16(data, 20));
    out->destination.vport = htons(ReadU16(data, 22));
    out->payload.assign(data + FrameHeaderSize, data + size);
    return true;
}

bool ConfigureEndpointLocked(BridgeState& state) {
    if (state.endpoint_configured) {
        return true;
    }
    std::string endpoint = "127.0.0.1:" + std::to_string(DefaultHelperPort);
    if (const char* configured = std::getenv("SHADPS4_LBP3_BRIDGE"); configured != nullptr) {
        endpoint = configured;
    }
    if (endpoint.empty() || endpoint == "off" || endpoint == "OFF") {
        return false;
    }
    const size_t separator = endpoint.rfind(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 == endpoint.size()) {
        LOG_ERROR(Lib_Net, "Invalid SHADPS4_LBP3_BRIDGE endpoint '{}'", endpoint);
        return false;
    }
    const std::string host = endpoint.substr(0, separator);
    char* end = nullptr;
    const long port = std::strtol(endpoint.c_str() + separator + 1, &end, 10);
    if (end == nullptr || *end != '\0' || port <= 0 || port > 65535) {
        LOG_ERROR(Lib_Net, "Invalid SHADPS4_LBP3_BRIDGE port in '{}'", endpoint);
        return false;
    }

    sockaddr_in helper{};
    helper.sin_family = AF_INET;
    helper.sin_port = htons(static_cast<u16>(port));
    if (inet_pton(AF_INET, host.c_str(), &helper.sin_addr) != 1 ||
        (ntohl(helper.sin_addr.s_addr) & 0xff000000u) != 0x7f000000u) {
        LOG_ERROR(Lib_Net, "LBP3 helper endpoint must be an IPv4 loopback address: '{}'", endpoint);
        return false;
    }
    state.helper = helper;
    state.endpoint_configured = true;
    return true;
}

bool OpenSocketLocked(BridgeState& state) {
    if (state.socket != InvalidSocket) {
        return true;
    }
    if (!ConfigureEndpointLocked(state)) {
        return false;
    }
    state.socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (state.socket == InvalidSocket) {
        LOG_ERROR(Lib_Net, "Failed to create LBP3 helper IPC socket");
        return false;
    }

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = 0;
    local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(state.socket, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
        LOG_ERROR(Lib_Net, "Failed to bind LBP3 helper IPC socket");
#ifdef _WIN32
        closesocket(state.socket);
#else
        close(state.socket);
#endif
        state.socket = InvalidSocket;
        return false;
    }

#ifdef _WIN32
    u_long nonblocking = 1;
    if (ioctlsocket(state.socket, FIONBIO, &nonblocking) != 0) {
        closesocket(state.socket);
        state.socket = InvalidSocket;
        return false;
    }
#else
    const int flags = fcntl(state.socket, F_GETFL, 0);
    if (flags < 0 || fcntl(state.socket, F_SETFL, flags | O_NONBLOCK) != 0) {
        close(state.socket);
        state.socket = InvalidSocket;
        return false;
    }
#endif
    if (!state.service_thread.joinable()) {
        state.service_thread =
            std::jthread([&state](std::stop_token stop_token) { ServiceLoop(state, stop_token); });
        LOG_INFO(Lib_Net, "Started independent LBP3 helper bridge service");
    }
    return true;
}

bool SameEndpoint(const sockaddr_in& left, const sockaddr_in& right) {
    return left.sin_family == right.sin_family && left.sin_port == right.sin_port &&
           left.sin_addr.s_addr == right.sin_addr.s_addr;
}

u32 VirtualAddrForOnlineIdImpl(std::string_view online_id) {
    u32 hash = 2166136261u;
    for (unsigned char value : online_id) {
        if (value >= 'A' && value <= 'Z') {
            value = static_cast<unsigned char>(value - 'A' + 'a');
        }
        hash ^= value;
        hash *= 16777619u;
    }
    u8 last = static_cast<u8>(hash);
    if (last == 0 || last == 255) {
        last ^= 0x5a;
        if (last == 0 || last == 255) {
            last = 1;
        }
    }
    const u32 host_order =
        (10u << 24) | ((hash >> 16) & 0xffu) << 16 | ((hash >> 8) & 0xffu) << 8 | last;
    return htonl(host_order);
}

void ExpireStaleConnectionLocked(BridgeState& state) {
    if (!state.connected || state.last_hello_ack.time_since_epoch().count() == 0 ||
        Clock::now() - state.last_hello_ack <= HelloAckTimeout) {
        return;
    }
    LOG_WARNING(Lib_Net, "LBP3 helper connection expired; waiting for a fresh HelloAck");
    state.connected = false;
    state.advertised_addr = 0;
    state.online_id.clear();
    for (auto& queue : state.queues) {
        queue.clear();
    }
}

void PumpLocked(BridgeState& state) {
    if (state.socket == InvalidSocket) {
        return;
    }
    std::array<u8, FrameHeaderSize + MaxPayloadSize> buffer{};
    for (;;) {
        sockaddr_in from{};
        socklen_t from_size = sizeof(from);
#ifdef _WIN32
        const int received = ::recvfrom(state.socket, reinterpret_cast<char*>(buffer.data()),
                                        static_cast<int>(buffer.size()), 0,
                                        reinterpret_cast<sockaddr*>(&from), &from_size);
        if (received == SOCKET_ERROR) {
            if (WSAGetLastError() != WSAEWOULDBLOCK) {
                LOG_WARNING(Lib_Net, "LBP3 helper IPC recvfrom failed: {}", WSAGetLastError());
            }
            break;
        }
#else
        const int received =
            static_cast<int>(::recvfrom(state.socket, buffer.data(), buffer.size(), 0,
                                        reinterpret_cast<sockaddr*>(&from), &from_size));
        if (received < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                LOG_WARNING(Lib_Net, "LBP3 helper IPC recvfrom failed: {}", errno);
            }
            break;
        }
#endif
        if (!SameEndpoint(from, state.helper)) {
            continue;
        }
        Frame frame{};
        if (!DecodeFrame(buffer.data(), static_cast<size_t>(received), &frame)) {
            LOG_WARNING(Lib_Net, "Ignored malformed LBP3 helper IPC frame ({} bytes)", received);
            continue;
        }
        if (frame.type == FrameTypeHelloAck) {
            const std::string online_id(frame.payload.begin(), frame.payload.end());
            if (frame.channel != Channel::Control || frame.flags != 0 || frame.source.addr == 0 ||
                frame.source.port != htons(DefaultGamePort) || !IsValidOnlineId(online_id) ||
                frame.source.addr != VirtualAddrForOnlineIdImpl(online_id)) {
                LOG_WARNING(Lib_Net, "Ignored invalid LBP3 helper identity");
                continue;
            }
            const bool changed = !state.connected || state.advertised_addr != frame.source.addr ||
                                 state.online_id != online_id;
            state.connected = true;
            state.advertised_addr = frame.source.addr;
            state.online_id = online_id;
            state.last_hello_ack = Clock::now();
            if (changed) {
                LOG_INFO(Lib_Net, "LBP3 helper connected: OnlineID='{}' virtual_addr={:#x}",
                         state.online_id, ntohl(state.advertised_addr));
            }
            continue;
        }
        if (frame.type != FrameTypeDatagram) {
            continue;
        }
        auto& queue = state.queues[static_cast<size_t>(frame.channel)];
        if (queue.size() == MaxQueuedDatagrams) {
            queue.pop_front();
        }
        queue.emplace_back(std::move(frame));
    }
}

bool SendRawLocked(BridgeState& state, const std::vector<u8>& encoded) {
    if (encoded.empty() || state.socket == InvalidSocket) {
        return false;
    }
#ifdef _WIN32
    const int sent =
        ::sendto(state.socket, reinterpret_cast<const char*>(encoded.data()),
                 static_cast<int>(encoded.size()), 0,
                 reinterpret_cast<const sockaddr*>(&state.helper), sizeof(state.helper));
    return sent == static_cast<int>(encoded.size());
#else
    const auto sent =
        ::sendto(state.socket, encoded.data(), encoded.size(), 0,
                 reinterpret_cast<const sockaddr*>(&state.helper), sizeof(state.helper));
    return sent == static_cast<ssize_t>(encoded.size());
#endif
}

bool SendHelloLocked(BridgeState& state) {
    const auto now = Clock::now();
    if (state.last_hello.time_since_epoch().count() != 0 &&
        now - state.last_hello < HelloInterval) {
        return false;
    }
    Frame hello{};
    hello.type = FrameTypeHello;
    hello.channel = Channel::Control;
    if (!SendRawLocked(state, EncodeFrame(hello))) {
        return false;
    }
    state.last_hello = now;
    return true;
}

void ServiceLoop(BridgeState& state, std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        {
            std::scoped_lock lock{state.mutex};
            PumpLocked(state);
            ExpireStaleConnectionLocked(state);
            SendHelloLocked(state);
        }
        std::this_thread::sleep_for(ServiceInterval);
    }
}

bool MatchesLocalEndpoint(const Frame& frame, const Endpoint* local) {
    if (local == nullptr || (local->port == 0 && local->vport == 0)) {
        return true;
    }
    return (local->port != 0 &&
            (frame.destination.port == local->port || frame.destination.vport == local->port)) ||
           (local->vport != 0 &&
            (frame.destination.vport == local->vport || frame.destination.port == local->vport));
}

} // namespace

bool IsSupportedTitle() {
    return Core::Lbp3Online::IsSupportedTitle();
}

bool EnsureConnected() {
    if (!IsSupportedTitle()) {
        return false;
    }
    BridgeState& state = State();
    std::unique_lock lock{state.mutex};
    if (!OpenSocketLocked(state)) {
        return false;
    }
    PumpLocked(state);
    ExpireStaleConnectionLocked(state);
    const bool hello_sent = SendHelloLocked(state);
    if (state.connected) {
        return true;
    }
    if (!hello_sent) {
        return false;
    }

    const auto deadline = Clock::now() + std::chrono::milliseconds(100);
    while (!state.connected && Clock::now() < deadline) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        lock.lock();
        PumpLocked(state);
        ExpireStaleConnectionLocked(state);
    }
    return state.connected;
}

bool IsConnected() {
    if (!IsSupportedTitle()) {
        return false;
    }
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    if (!OpenSocketLocked(state)) {
        return false;
    }
    PumpLocked(state);
    ExpireStaleConnectionLocked(state);
    SendHelloLocked(state);
    return state.connected;
}

u32 AdvertisedAddr() {
    EnsureConnected();
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    return state.advertised_addr;
}

u16 ConfiguredPort() {
    return htons(DefaultGamePort);
}

std::string OnlineId() {
    EnsureConnected();
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    return state.online_id;
}

u32 VirtualAddrForOnlineId(std::string_view online_id) {
    return VirtualAddrForOnlineIdImpl(online_id);
}

bool ResolvePeer(std::string_view online_id, u32* out_addr, u16* out_port) {
    if (!IsSupportedTitle() || out_addr == nullptr || out_port == nullptr ||
        !IsValidOnlineId(online_id)) {
        return false;
    }
    *out_addr = VirtualAddrForOnlineId(online_id);
    *out_port = ConfiguredPort();
    return true;
}

void ObserveMatchingRequest(std::string_view body) {
    if (!IsSupportedTitle()) {
        return;
    }
    if (body.contains("UpdateMyPlayerData") || body.contains("CreateRoom") ||
        body.contains("UpdatePlayersInRoom")) {
        State().update_my_player_data_seen.store(true, std::memory_order_release);
    }
}

namespace {

bool HasBytes(uintptr_t address, std::initializer_list<u8> expected) {
    const auto* bytes = reinterpret_cast<const u8*>(address);
    return std::equal(expected.begin(), expected.end(), bytes);
}

bool ValidateFindBestRoomGuestCode(uintptr_t base) {
    return HasBytes(base + MatchingDispatcherOffset,
                    {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56}) &&
           HasBytes(base + MatchingDispatcherOffset + 0x18a,
                    {0x49, 0x8b, 0xbc, 0x24, 0x30, 0x02, 0x00, 0x00, 0x8b, 0x1f}) &&
           HasBytes(base + FindBestRoomBuilderOffset,
                    {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55}) &&
           HasBytes(base + VectorPushU64Offset,
                    {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53});
}

bool HelperRosterRequestsJoin(const std::string& local_online_id) {
#ifndef LBP3_BRIDGE_WITH_HTTPLIB
    return false;
#else
    httplib::Client client("http://127.0.0.1:18063");
    client.set_connection_timeout(std::chrono::milliseconds(50));
    client.set_read_timeout(std::chrono::milliseconds(100));
    client.set_write_timeout(std::chrono::milliseconds(100));
    const auto response = client.Get("/status");
    if (!response || response->status != 200) {
        return false;
    }

    try {
        const auto status = nlohmann::json::parse(response->body);
        if (status.value("role", std::string{}) != "member") {
            return false;
        }
        const auto& roster = status.at("roster");
        if (!roster.is_array() || roster.size() < 2 || !roster.front().is_string() ||
            roster.front().get<std::string>() == local_online_id) {
            return false;
        }
        return std::any_of(roster.begin(), roster.end(), [&](const auto& entry) {
            return entry.is_string() && entry.template get<std::string>() == local_online_id;
        });
    } catch (const std::exception& error) {
        LOG_WARNING(Lib_Net, "LBP3 helper /status could not arm FindBestRoom: {}", error.what());
        return false;
    }
#endif
}

} // namespace

void MaybeQueueFindBestRoom() {
#if !defined(ARCH_X86_64)
    return;
#else
    BridgeState& state = State();
    if (!IsSupportedTitle() || state.find_best_room_queued.load(std::memory_order_acquire) ||
        state.find_best_room_signatures_rejected.load(std::memory_order_relaxed)) {
        return;
    }

    std::unique_lock join_lock{state.find_best_room_mutex, std::try_to_lock};
    if (!join_lock.owns_lock() || state.find_best_room_queued.load(std::memory_order_acquire)) {
        return;
    }

    const uintptr_t base = MemoryPatcher::g_eboot_address;
    if (base == 0) {
        return;
    }
    if (!ValidateFindBestRoomGuestCode(base)) {
        if (!state.find_best_room_signatures_rejected.exchange(true)) {
            LOG_ERROR(Lib_Net,
                      "LBP3 FindBestRoom hook disabled: CUSA00063 01.26 guest signatures differ");
        }
        return;
    }

    const auto now = Clock::now();
    constexpr auto HelperStatusInterval = std::chrono::seconds(1);
    if (state.helper_roster_last_check.time_since_epoch().count() == 0 ||
        now - state.helper_roster_last_check >= HelperStatusInterval) {
        state.helper_roster_last_check = now;
        state.helper_roster_local_online_id = OnlineId();
        state.helper_roster_requests_join =
            !state.helper_roster_local_online_id.empty() &&
            HelperRosterRequestsJoin(state.helper_roster_local_online_id);
        if (!state.helper_roster_requests_join) {
            state.helper_roster_first_seen = {};
        } else if (state.helper_roster_first_seen.time_since_epoch().count() == 0) {
            state.helper_roster_first_seen = now;
            LOG_INFO(Lib_Net, "LBP3 helper host-first member roster observed; arming FindBestRoom");
        }
    }
    if (!state.helper_roster_requests_join) {
        return;
    }

    const std::string& local_online_id = state.helper_roster_local_online_id;
    if (!state.update_my_player_data_seen.load(std::memory_order_acquire) &&
        now - state.helper_roster_first_seen < std::chrono::seconds(12)) {
        return;
    }

    const uintptr_t manager = base + OnlineManagerOffset;
    const uintptr_t matching =
        *reinterpret_cast<const uintptr_t*>(manager + MatchingObjectPointerOffset);
    if (matching == 0 || matching < 0x100000000ull || matching >= 0x1000000000ull ||
        *reinterpret_cast<const u32*>(matching + MatchingActiveStateOffset) != 0 ||
        *reinterpret_cast<const u32*>(matching + MatchingBlockingStateOffset) != 0) {
        return;
    }

    const u32 size = *reinterpret_cast<const u32*>(matching + MatchingQueueSizeOffset);
    const u32 capacity = *reinterpret_cast<const u32*>(matching + MatchingQueueCapacityOffset);
    const uintptr_t data = *reinterpret_cast<const uintptr_t*>(matching + MatchingQueueOffset);
    if (size != 0 || size > capacity || (capacity != 0 && data == 0)) {
        return;
    }

    using PushU64 = PS4_SYSV_ABI void (*)(void*, const u64*);
    const auto push = reinterpret_cast<PushU64>(base + VectorPushU64Offset);
    const u64 selector = FindBestRoomSelector;
    push(reinterpret_cast<void*>(matching + MatchingQueueOffset), &selector);
    state.find_best_room_queued.store(true, std::memory_order_release);
    LOG_CRITICAL(Lib_Net,
                 "LBP3 FindBestRoom selector queued (selector={} local='{}' matching={:#x})",
                 selector, local_online_id, matching);
#endif
}

int Send(Channel channel, const void* data, u32 len, const Endpoint& source,
         const Endpoint& destination) {
    if ((data == nullptr && len != 0) || len > MaxPayloadSize || !EnsureConnected()) {
        return -1;
    }
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    PumpLocked(state);
    SendHelloLocked(state);
    Frame frame{};
    frame.type = FrameTypeDatagram;
    frame.channel = channel;
    frame.source = source;
    frame.destination = destination;
    if (frame.source.addr == 0) {
        frame.source.addr = state.advertised_addr;
    }
    const auto* bytes = static_cast<const u8*>(data);
    if (len != 0) {
        frame.payload.assign(bytes, bytes + len);
    }
    return SendRawLocked(state, EncodeFrame(frame)) ? static_cast<int>(len) : -1;
}

int Receive(Channel channel, void* data, u32 len, const Endpoint* local, Endpoint* source) {
    if ((data == nullptr && len != 0) || !EnsureConnected()) {
        return -1;
    }
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    PumpLocked(state);
    auto& queue = state.queues[static_cast<size_t>(channel)];
    const auto found = std::find_if(queue.begin(), queue.end(), [&](const Frame& frame) {
        return MatchesLocalEndpoint(frame, local);
    });
    if (found == queue.end()) {
        return -1;
    }
    Frame frame = std::move(*found);
    queue.erase(found);
    const size_t received = std::min<size_t>(len, frame.payload.size());
    if (received != 0) {
        std::memcpy(data, frame.payload.data(), received);
    }
    if (source != nullptr) {
        *source = frame.source;
    }
    return static_cast<int>(received);
}

bool HasPending(Channel channel, const Endpoint* local) {
    if (!IsConnected()) {
        return false;
    }
    BridgeState& state = State();
    std::scoped_lock lock{state.mutex};
    PumpLocked(state);
    const auto& queue = state.queues[static_cast<size_t>(channel)];
    return std::any_of(queue.begin(), queue.end(),
                       [&](const Frame& frame) { return MatchesLocalEndpoint(frame, local); });
}

} // namespace Libraries::Net::Lbp3OnlineBridge
