// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>

#include <common/assert.h>
#include <common/logging/log.h>
#include <common/singleton.h>
#include "core/libraries/kernel/kernel.h"
#include "lbp3_online_bridge.h"
#include "net.h"
#include "net_error.h"
#include "net_util.h"
#include "sockets.h"

namespace Libraries::Net {

namespace {

std::mutex p2p_registry_mutex;
std::map<u16, std::weak_ptr<P2PSocket>> p2p_registry;

int SetP2PError(int error) {
    *Libraries::Kernel::__Error() = error;
    return -1;
}

void RemoveSocketFromRegistry(P2PSocket* socket) {
    for (auto it = p2p_registry.begin(); it != p2p_registry.end();) {
        const auto registered = it->second.lock();
        if (!registered || registered.get() == socket) {
            it = p2p_registry.erase(it);
        } else {
            ++it;
        }
    }
}

bool IsLoopbackAddress(u32 address) {
    return (ntohl(address) & 0xff000000) == 0x7f000000;
}

} // namespace

u16 GetP2PConfiguredPort() {
    return Lbp3OnlineBridge::IsSupportedTitle() ? Lbp3OnlineBridge::ConfiguredPort() : 0;
}

u32 GetP2PAdvertisedAddr() {
    if (Lbp3OnlineBridge::IsSupportedTitle() && Lbp3OnlineBridge::EnsureConnected()) {
        return Lbp3OnlineBridge::AdvertisedAddr();
    }
    auto* netinfo = Common::Singleton<NetUtil::NetUtilInternal>::Instance();
    if (netinfo->RetrieveIp()) {
        const u32 address = inet_addr(netinfo->GetIp().c_str());
        if (address != INADDR_NONE) {
            return address;
        }
    }
    return htonl(INADDR_LOOPBACK);
}

bool EnsureP2PTransport() {
    return !Lbp3OnlineBridge::IsSupportedTitle() || Lbp3OnlineBridge::EnsureConnected();
}

bool P2PTransportIsReady() {
    return !Lbp3OnlineBridge::IsSupportedTitle() || Lbp3OnlineBridge::IsConnected();
}

int P2PSignalingSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    const Lbp3OnlineBridge::Endpoint source{
        .addr = GetP2PAdvertisedAddr(),
        .port = GetP2PConfiguredPort(),
    };
    const Lbp3OnlineBridge::Endpoint destination{.addr = dest_addr, .port = dest_port};
    return Lbp3OnlineBridge::Send(Lbp3OnlineBridge::Channel::Signaling, data, len, source,
                                  destination);
}

int P2PSignalingRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    Lbp3OnlineBridge::Endpoint source{};
    const int received =
        Lbp3OnlineBridge::Receive(Lbp3OnlineBridge::Channel::Signaling, buf, len, nullptr, &source);
    if (received >= 0) {
        if (from_addr != nullptr) {
            *from_addr = source.addr;
        }
        if (from_port != nullptr) {
            *from_port = source.port != 0 ? source.port : source.vport;
        }
    }
    return received;
}

int P2PControlSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    const Lbp3OnlineBridge::Endpoint source{
        .addr = GetP2PAdvertisedAddr(),
        .port = GetP2PConfiguredPort(),
    };
    const Lbp3OnlineBridge::Endpoint destination{.addr = dest_addr, .port = dest_port};
    return Lbp3OnlineBridge::Send(Lbp3OnlineBridge::Channel::Control, data, len, source,
                                  destination);
}

int P2PControlRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    Lbp3OnlineBridge::Endpoint source{};
    const int received =
        Lbp3OnlineBridge::Receive(Lbp3OnlineBridge::Channel::Control, buf, len, nullptr, &source);
    if (received >= 0) {
        if (from_addr != nullptr) {
            *from_addr = source.addr;
        }
        if (from_port != nullptr) {
            *from_port = source.port != 0 ? source.port : source.vport;
        }
    }
    return received;
}

bool P2PResolvePeer(std::string_view online_id, u32* out_addr, u16* out_port) {
    return Lbp3OnlineBridge::ResolvePeer(online_id, out_addr, out_port);
}

int P2PMatching2SendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    return -1;
}

int P2PMatching2RecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    return -1;
}

P2PSocket::~P2PSocket() {
    Unregister();
}

void P2PSocket::Unregister() {
    std::scoped_lock lock{p2p_registry_mutex};
    RemoveSocketFromRegistry(this);
    is_bound = false;
}

int P2PSocket::Close() {
    {
        std::scoped_lock lock{p2p_registry_mutex};
        RemoveSocketFromRegistry(this);
        is_bound = false;
        is_connected = false;
        peer_addr = {};
    }
    {
        std::scoped_lock lock{local_receive_mutex};
        is_closed = true;
        local_receive_queue.clear();
    }
    local_receive_cv.notify_all();
    return inner.Close();
}

u32 P2PSocket::GetPendingEvents(u32 requested_events) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return 0;
    }

    Lbp3OnlineBridge::Endpoint local{};
    {
        std::scoped_lock lock{p2p_registry_mutex};
        if (is_bound) {
            local = {
                .addr = bound_addr.sin_addr,
                .port = bound_addr.sin_port,
                .vport = bound_addr.sin_vport,
            };
        }
    }
    u32 pending_events = requested_events & ORBIS_NET_EPOLLOUT;
    {
        std::scoped_lock lock{local_receive_mutex};
        if (!local_receive_queue.empty()) {
            pending_events |= requested_events & ORBIS_NET_EPOLLIN;
        }
        if (is_closed) {
            pending_events |= (requested_events & ORBIS_NET_EPOLLIN) | ORBIS_NET_EPOLLHUP;
        }
    }
    if ((requested_events & ORBIS_NET_EPOLLIN) != 0 &&
        Lbp3OnlineBridge::HasPending(Lbp3OnlineBridge::Channel::Game, &local)) {
        pending_events |= ORBIS_NET_EPOLLIN;
    }
    return pending_events;
}

int P2PSocket::Bind(const OrbisNetSockaddr* addr, u32 addrlen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.Bind(addr, addrlen);
    }
    if (addr == nullptr || addrlen < sizeof(OrbisNetSockaddrIn)) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    OrbisNetSockaddrIn new_addr{};
    std::memcpy(&new_addr, addr, sizeof(new_addr));
    if (new_addr.sin_family != ORBIS_NET_AF_INET) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    const auto self = weak_from_this();
    if (self.expired()) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    {
        std::scoped_lock lock{p2p_registry_mutex};
        RemoveSocketFromRegistry(this);
        bound_addr = new_addr;
        is_bound = true;
        if (bound_addr.sin_port != 0) {
            p2p_registry[bound_addr.sin_port] = self;
        }
        if (bound_addr.sin_vport != 0 && bound_addr.sin_vport != bound_addr.sin_port) {
            p2p_registry[bound_addr.sin_vport] = self;
        }
    }
    LOG_DEBUG(Lib_Net, "Bound local P2P datagram socket to port {:#x}, vport {:#x}",
              ntohs(bound_addr.sin_port), ntohs(bound_addr.sin_vport));
    if (Lbp3OnlineBridge::IsSupportedTitle() && !Lbp3OnlineBridge::EnsureConnected()) {
        LOG_WARNING(Lib_Net, "LBP3 P2P socket is bound, but the local helper is not reachable yet");
    }
    return 0;
}

int P2PSocket::SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                          u32 tolen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.SendPacket(msg, len, flags, to, tolen);
    }
    if (msg == nullptr && len != 0) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    OrbisNetSockaddrIn destination_addr{};
    if (to != nullptr) {
        if (tolen < sizeof(OrbisNetSockaddrIn)) {
            return SetP2PError(ORBIS_NET_EINVAL);
        }
        std::memcpy(&destination_addr, to, sizeof(destination_addr));
    } else {
        std::scoped_lock lock{p2p_registry_mutex};
        if (!is_connected) {
            return SetP2PError(ORBIS_NET_EDESTADDRREQ);
        }
        destination_addr = peer_addr;
    }
    if (destination_addr.sin_family != ORBIS_NET_AF_INET) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    const u32 advertised_addr = GetP2PAdvertisedAddr();
    OrbisNetSockaddrIn source_addr{};
    bool source_is_bound = false;
    std::shared_ptr<P2PSocket> destination;
    {
        std::scoped_lock lock{p2p_registry_mutex};
        source_addr = bound_addr;
        source_is_bound = is_bound;

        const auto find_destination = [&](u16 port) -> std::shared_ptr<P2PSocket> {
            if (port == 0) {
                return {};
            }
            const auto it = p2p_registry.find(port);
            if (it == p2p_registry.end()) {
                return {};
            }
            auto candidate = it->second.lock();
            if (!candidate) {
                p2p_registry.erase(it);
                return {};
            }

            const u32 address = destination_addr.sin_addr;
            const bool is_local =
                address == htonl(INADDR_ANY) || IsLoopbackAddress(address) ||
                address == advertised_addr ||
                (is_bound && bound_addr.sin_addr != htonl(INADDR_ANY) &&
                 address == bound_addr.sin_addr) ||
                (candidate->is_bound && candidate->bound_addr.sin_addr != htonl(INADDR_ANY) &&
                 address == candidate->bound_addr.sin_addr);
            return is_local ? candidate : std::shared_ptr<P2PSocket>{};
        };

        destination = find_destination(destination_addr.sin_vport);
        if (!destination && destination_addr.sin_port != destination_addr.sin_vport) {
            destination = find_destination(destination_addr.sin_port);
        }
    }

    if (!destination) {
        if (!source_is_bound) {
            source_addr = {
                .sin_len = sizeof(OrbisNetSockaddrIn),
                .sin_family = ORBIS_NET_AF_INET,
                .sin_addr = advertised_addr,
            };
        } else if (source_addr.sin_addr == htonl(INADDR_ANY)) {
            source_addr.sin_addr = advertised_addr;
        }
        const Lbp3OnlineBridge::Endpoint source{
            .addr = source_addr.sin_addr,
            .port = source_addr.sin_port,
            .vport = source_addr.sin_vport,
        };
        const Lbp3OnlineBridge::Endpoint remote{
            .addr = destination_addr.sin_addr,
            .port = destination_addr.sin_port,
            .vport = destination_addr.sin_vport,
        };
        const int sent =
            Lbp3OnlineBridge::Send(Lbp3OnlineBridge::Channel::Game, msg, len, source, remote);
        if (sent >= 0) {
            LOG_DEBUG(Lib_Net,
                      "Relayed {} byte LBP3 P2P datagram to {:#x}, port {:#x}, vport {:#x}", len,
                      ntohl(destination_addr.sin_addr), ntohs(destination_addr.sin_port),
                      ntohs(destination_addr.sin_vport));
            return sent;
        }
        LOG_ERROR(Lib_Net,
                  "No local P2P datagram destination for address {:#x}, port {:#x}, vport {:#x}",
                  ntohl(destination_addr.sin_addr), ntohs(destination_addr.sin_port),
                  ntohs(destination_addr.sin_vport));
        return SetP2PError(ORBIS_NET_EAGAIN);
    }

    if (!source_is_bound) {
        source_addr = {
            .sin_len = sizeof(OrbisNetSockaddrIn),
            .sin_family = ORBIS_NET_AF_INET,
            .sin_addr = htonl(INADDR_LOOPBACK),
        };
    } else if (source_addr.sin_addr == htonl(INADDR_ANY)) {
        source_addr.sin_addr = destination_addr.sin_addr == htonl(INADDR_ANY)
                                   ? advertised_addr
                                   : destination_addr.sin_addr;
    }
    source_addr.sin_len = sizeof(OrbisNetSockaddrIn);
    source_addr.sin_family = ORBIS_NET_AF_INET;

    LocalDatagram packet{};
    packet.data.resize(len);
    if (len != 0) {
        std::memcpy(packet.data.data(), msg, len);
    }
    packet.source = source_addr;

    {
        std::scoped_lock lock{destination->local_receive_mutex};
        if (destination->is_closed) {
            return SetP2PError(ORBIS_NET_EAGAIN);
        }
        if (destination->local_receive_queue.size() == MaxQueuedDatagrams) {
            destination->local_receive_queue.pop_front();
        }
        destination->local_receive_queue.emplace_back(std::move(packet));
    }
    destination->local_receive_cv.notify_one();
    LOG_DEBUG(Lib_Net, "Delivered {} byte local P2P datagram to port {:#x}, vport {:#x}", len,
              ntohs(destination_addr.sin_port), ntohs(destination_addr.sin_vport));
    return static_cast<int>(len);
}

int P2PSocket::SendMessage(const OrbisNetMsghdr* msg, int flags) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.SendMessage(msg, flags);
    }
    if (msg == nullptr || msg->msg_iovlen < 0 ||
        (msg->msg_iovlen != 0 && msg->msg_iov == nullptr)) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    size_t total = 0;
    for (int i = 0; i < msg->msg_iovlen; ++i) {
        const auto& iov = msg->msg_iov[i];
        if (iov.iov_base == nullptr && iov.iov_len != 0) {
            return SetP2PError(ORBIS_NET_EFAULT);
        }
        if (iov.iov_len > std::numeric_limits<u32>::max() - total) {
            return SetP2PError(ORBIS_NET_EMSGSIZE);
        }
        total += static_cast<size_t>(iov.iov_len);
    }

    std::vector<u8> payload(total);
    size_t cursor = 0;
    for (int i = 0; i < msg->msg_iovlen; ++i) {
        const auto& iov = msg->msg_iov[i];
        if (iov.iov_len != 0) {
            std::memcpy(payload.data() + cursor, iov.iov_base, static_cast<size_t>(iov.iov_len));
            cursor += static_cast<size_t>(iov.iov_len);
        }
    }

    const auto* destination = static_cast<const OrbisNetSockaddr*>(msg->msg_name);
    return SendPacket(payload.empty() ? nullptr : payload.data(), static_cast<u32>(payload.size()),
                      flags, destination, destination != nullptr ? msg->msg_namelen : 0);
}

int P2PSocket::ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from, u32* fromlen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.ReceivePacket(buf, len, flags, from, fromlen);
    }
    if (buf == nullptr && len != 0) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    LocalDatagram packet{};
    bool have_local_packet = false;
    {
        std::scoped_lock lock{local_receive_mutex};
        if (!local_receive_queue.empty()) {
            packet = std::move(local_receive_queue.front());
            local_receive_queue.pop_front();
            have_local_packet = true;
        }
    }

    if (have_local_packet) {
        const size_t received = std::min<size_t>(len, packet.data.size());
        if (received != 0) {
            std::memcpy(buf, packet.data.data(), received);
        }
        if (fromlen != nullptr) {
            const u32 available = *fromlen;
            *fromlen = sizeof(packet.source);
            if (from != nullptr && available != 0) {
                std::memcpy(from, &packet.source, std::min<u32>(available, sizeof(packet.source)));
            }
        }
        return static_cast<int>(received);
    }

    Lbp3OnlineBridge::Endpoint local{};
    {
        std::scoped_lock lock{p2p_registry_mutex};
        if (is_bound) {
            local = {
                .addr = bound_addr.sin_addr,
                .port = bound_addr.sin_port,
                .vport = bound_addr.sin_vport,
            };
        }
    }
    Lbp3OnlineBridge::Endpoint remote{};
    const int received =
        Lbp3OnlineBridge::Receive(Lbp3OnlineBridge::Channel::Game, buf, len, &local, &remote);
    if (received < 0) {
        return SetP2PError(ORBIS_NET_EAGAIN);
    }
    if (fromlen != nullptr) {
        OrbisNetSockaddrIn source{
            .sin_len = sizeof(OrbisNetSockaddrIn),
            .sin_family = ORBIS_NET_AF_INET,
            .sin_port = remote.port,
            .sin_addr = remote.addr,
            .sin_vport = remote.vport,
        };
        const u32 available = *fromlen;
        *fromlen = sizeof(source);
        if (from != nullptr && available != 0) {
            std::memcpy(from, &source, std::min<u32>(available, sizeof(source)));
        }
    }
    return received;
}

int P2PSocket::ReceiveMessage(OrbisNetMsghdr* msg, int flags) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.ReceiveMessage(msg, flags);
    }
    if (msg == nullptr || msg->msg_iovlen < 0 ||
        (msg->msg_iovlen != 0 && msg->msg_iov == nullptr)) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    size_t capacity = 0;
    for (int i = 0; i < msg->msg_iovlen; ++i) {
        const auto& iov = msg->msg_iov[i];
        if (iov.iov_base == nullptr && iov.iov_len != 0) {
            return SetP2PError(ORBIS_NET_EFAULT);
        }
        if (iov.iov_len > std::numeric_limits<u32>::max() - capacity) {
            return SetP2PError(ORBIS_NET_EMSGSIZE);
        }
        capacity += static_cast<size_t>(iov.iov_len);
    }

    std::vector<u8> payload(capacity);
    u32 source_len = msg->msg_name != nullptr ? msg->msg_namelen : 0;
    const int received =
        ReceivePacket(payload.empty() ? nullptr : payload.data(), static_cast<u32>(payload.size()),
                      flags, static_cast<OrbisNetSockaddr*>(msg->msg_name),
                      msg->msg_name != nullptr ? &source_len : nullptr);
    if (received < 0) {
        return received;
    }
    if (msg->msg_name != nullptr) {
        msg->msg_namelen = source_len;
    }
    msg->msg_flags = 0;

    size_t cursor = 0;
    size_t remaining = static_cast<size_t>(received);
    for (int i = 0; i < msg->msg_iovlen && remaining != 0; ++i) {
        const size_t copied = std::min<size_t>(remaining, msg->msg_iov[i].iov_len);
        if (copied != 0) {
            std::memcpy(msg->msg_iov[i].iov_base, payload.data() + cursor, copied);
            cursor += copied;
            remaining -= copied;
        }
    }
    return received;
}

int P2PSocket::Connect(const OrbisNetSockaddr* addr, u32 namelen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.Connect(addr, namelen);
    }
    if (addr == nullptr || namelen < sizeof(OrbisNetSockaddrIn)) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    OrbisNetSockaddrIn destination{};
    std::memcpy(&destination, addr, sizeof(destination));
    if (destination.sin_family != ORBIS_NET_AF_INET) {
        return SetP2PError(ORBIS_NET_EAFNOSUPPORT);
    }
    {
        std::scoped_lock lock{p2p_registry_mutex};
        peer_addr = destination;
        is_connected = true;
    }
    LOG_DEBUG(Lib_Net, "Connected virtual P2P datagram socket to {:#x}, port {:#x}, vport {:#x}",
              ntohl(destination.sin_addr), ntohs(destination.sin_port),
              ntohs(destination.sin_vport));
    return 0;
}

int P2PSocket::GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.GetSocketAddress(name, namelen);
    }
    if (name == nullptr || namelen == nullptr) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    OrbisNetSockaddrIn address{};
    bool socket_is_bound = false;
    {
        std::scoped_lock lock{p2p_registry_mutex};
        socket_is_bound = is_bound;
        if (socket_is_bound) {
            address = bound_addr;
        }
    }
    if (!socket_is_bound) {
        return inner.GetSocketAddress(name, namelen);
    }

    const u32 available = *namelen;
    *namelen = sizeof(address);
    if (available != 0) {
        std::memcpy(name, &address, std::min<u32>(available, sizeof(address)));
    }
    return 0;
}

int P2PSocket::GetPeerName(OrbisNetSockaddr* addr, u32* namelen) {
    if (socket_type != ORBIS_NET_SOCK_DGRAM_P2P) {
        return inner.GetPeerName(addr, namelen);
    }
    if (addr == nullptr || namelen == nullptr) {
        return SetP2PError(ORBIS_NET_EINVAL);
    }

    OrbisNetSockaddrIn peer{};
    {
        std::scoped_lock lock{p2p_registry_mutex};
        if (!is_connected) {
            return SetP2PError(ORBIS_NET_ENOTCONN);
        }
        peer = peer_addr;
    }
    const u32 available = *namelen;
    *namelen = sizeof(peer);
    if (available != 0) {
        std::memcpy(addr, &peer, std::min<u32>(available, sizeof(peer)));
    }
    return 0;
}

} // namespace Libraries::Net
