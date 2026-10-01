// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include <core/user_settings.h>
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "core/emulator_settings.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/lbp3_online_bridge.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/np/np_manager.h"
#include "core/tls.h"
#include "core/user_manager.h"
#include "np_handler.h"

namespace Libraries::Np::NpManager {

static bool g_shadnet_enabled = false;
static s32 g_firmware_version = -1;
static s32 g_active_requests = 0;
static std::mutex g_request_mutex;

static std::mutex g_lookup_title_ctx_mutex;
static std::map<s32, OrbisNpId> g_lookup_title_ctxs;
static s32 g_next_lookup_title_ctx_id = 1;

static std::map<std::string, std::function<void()>> g_np_callbacks;
static std::mutex g_np_callbacks_mutex;
static std::mutex g_np_state_events_mutex;
static std::mutex g_np_state_callbacks_mutex;

constexpr s32 ORBIS_NP_STATE_CALLBACK_MAX = 8;

s32 PS4_SYSV_ABI sceNpLookupCreateTitleCtx(const OrbisNpId* self_np_id) {
    if (self_np_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lock{g_lookup_title_ctx_mutex};
    // Context IDs are positive handles.  Skip any live ID if the counter ever wraps during a
    // long-running session; zero and negative values are errors in the guest API.
    for (u32 attempts = 0; attempts < 0x7fffffff; ++attempts) {
        if (g_next_lookup_title_ctx_id <= 0) {
            g_next_lookup_title_ctx_id = 1;
        }
        const s32 id = g_next_lookup_title_ctx_id++;
        if (!g_lookup_title_ctxs.contains(id)) {
            g_lookup_title_ctxs.emplace(id, *self_np_id);
            LOG_INFO(Lib_NpManager, "sceNpLookupCreateTitleCtx: id={}", id);
            return id;
        }
    }
    return ORBIS_NP_ERROR_OUT_OF_MEMORY;
}

s32 PS4_SYSV_ABI sceNpLookupDeleteTitleCtx(s32 title_ctx_id) {
    std::scoped_lock lock{g_lookup_title_ctx_mutex};
    if (title_ctx_id <= 0 || g_lookup_title_ctxs.erase(title_ctx_id) == 0) {
        return ORBIS_NP_ERROR_INVALID_ID;
    }
    LOG_INFO(Lib_NpManager, "sceNpLookupDeleteTitleCtx: id={}", title_ctx_id);
    return ORBIS_OK;
}

// Internal types for storing request-related information
enum class NpRequestState {
    None = 0,
    Ready = 1,
    Aborted = 2,
    Complete = 3,
};

struct NpRequest {
    NpRequestState state;
    bool async;
    s32 result;
};

static void FillCountryCodeFromProfile(Libraries::UserService::OrbisUserServiceUserId user_id,
                                       OrbisNpCountryCode* out) {
    std::memset(out, 0, sizeof(OrbisNpCountryCode));
    const User* u = UserManagement.GetUserByID(user_id);
    const std::string cfg = u ? u->np_country : std::string();
    const bool ok = cfg.size() == 2 && std::isalpha(static_cast<unsigned char>(cfg[0])) &&
                    std::isalpha(static_cast<unsigned char>(cfg[1]));
    const char* src = ok ? cfg.c_str() : "us";
    std::memcpy(out->country_code, src, 2);
}

static void FillLanguageCodeFromProfile(Libraries::UserService::OrbisUserServiceUserId user_id,
                                        OrbisNpLanguageCode* out) {
    std::memset(out, 0, sizeof(OrbisNpLanguageCode));
    const User* u = UserManagement.GetUserByID(user_id);
    const std::string cfg = u ? u->np_language : std::string();
    const bool ok = cfg.size() == 2 && std::isalpha(static_cast<unsigned char>(cfg[0])) &&
                    std::isalpha(static_cast<unsigned char>(cfg[1]));
    const char* src = ok ? cfg.c_str() : "en";
    std::memcpy(out->code, src, 2);
}

static s8 GetAgeFromProfile(Libraries::UserService::OrbisUserServiceUserId user_id) {
    const User* u = UserManagement.GetUserByID(user_id);
    const int v = u ? static_cast<int>(u->np_age) : 0;
    if (v <= 0 || v > 127) {
        return 13;
    }
    return static_cast<s8>(v);
}

static void FillDateOfBirthFromProfile(Libraries::UserService::OrbisUserServiceUserId user_id,
                                       OrbisNpDate* out) {
    const User* u = UserManagement.GetUserByID(user_id);
    const std::string s = u ? u->np_date_of_birth : std::string();

    int y = 0, m = 0, d = 0;
    bool ok = s.size() == 10 && s[4] == '-' && s[7] == '-';
    if (ok) {
        auto is_digit_at = [&](size_t i) { return std::isdigit(static_cast<unsigned char>(s[i])); };
        for (size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u}) {
            if (!is_digit_at(i)) {
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        y = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
        m = (s[5] - '0') * 10 + (s[6] - '0');
        d = (s[8] - '0') * 10 + (s[9] - '0');
        ok = y >= 1900 && y <= 2100 && m >= 1 && m <= 12 && d >= 1 && d <= 31;
    }

    out->year = ok ? static_cast<u16>(y) : 2000;
    out->month = ok ? static_cast<u16>(m) : 1;
    out->day = ok ? static_cast<u16>(d) : 1;
}

static bool GetLbp3HelperOnlineId(Libraries::UserService::OrbisUserServiceUserId user_id,
                                  std::string* online_id) {
    if (online_id == nullptr || !Net::Lbp3OnlineBridge::IsSupportedTitle()) {
        return false;
    }
    const User* user = UserManagement.GetUserByID(user_id);
    if (user == nullptr || !user->logged_in || !Net::Lbp3OnlineBridge::EnsureConnected()) {
        return false;
    }
    *online_id = Net::Lbp3OnlineBridge::OnlineId();
    return !online_id->empty();
}

static bool IsLbp3HelperAvailable(Libraries::UserService::OrbisUserServiceUserId user_id) {
    std::string online_id;
    return GetLbp3HelperOnlineId(user_id, &online_id);
}

static bool IsNpUserOnline(Libraries::UserService::OrbisUserServiceUserId user_id) {
    return IsLbp3HelperAvailable(user_id) ||
           (g_shadnet_enabled && Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id));
}

static s32 FindLbp3HelperUserByOnlineId(const OrbisNpOnlineId& online_id) {
    for (const User* user : UserManagement.GetLoggedInUsers()) {
        if (user == nullptr) {
            continue;
        }
        std::string helper_online_id;
        if (!GetLbp3HelperOnlineId(user->user_id, &helper_online_id)) {
            continue;
        }
        OrbisNpOnlineId helper_id{};
        SetNpOnlineId(helper_id, helper_online_id);
        if (std::memcmp(helper_id.data, online_id.data, sizeof(helper_id.data)) == 0) {
            return user->user_id;
        }
    }
    return -1;
}

static u64 Lbp3HelperAccountId(Libraries::UserService::OrbisUserServiceUserId user_id) {
    std::string online_id;
    if (!GetLbp3HelperOnlineId(user_id, &online_id)) {
        return 0;
    }
    u64 hash = 1469598103934665603ULL;
    for (unsigned char value : online_id) {
        if (value >= 'A' && value <= 'Z') {
            value = static_cast<unsigned char>(value - 'A' + 'a');
        }
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash != 0 ? hash : 1;
}

static std::vector<NpRequest> g_requests;

static s32 CreateNpRequest(bool async) {
    std::scoped_lock lk{g_request_mutex};

    if (g_active_requests == ORBIS_NP_MANAGER_REQUEST_LIMIT) {
        return ORBIS_NP_ERROR_REQUEST_MAX;
    }

    s32 req_index = 0;
    while (req_index < static_cast<s32>(g_requests.size())) {
        // Find first nonexistant request
        if (g_requests[req_index].state == NpRequestState::None) {
            // There is no request at this index, set the index to ready then break.
            g_requests[req_index].state = NpRequestState::Ready;
            g_requests[req_index].async = async;
            break;
        }
        req_index++;
    }

    if (req_index == static_cast<s32>(g_requests.size())) {
        // There are no requests to replace.
        NpRequest new_request{NpRequestState::Ready, async, 0};
        g_requests.emplace_back(new_request);
    }

    // Offset by one, first returned ID is 0x20000001
    g_active_requests++;
    return req_index + ORBIS_NP_MANAGER_REQUEST_ID_OFFSET + 1;
}

// Validate a request ID and return the NpRequest*, or nullptr + error code.
// Writes the error code to *out_err when returning nullptr.
static NpRequest* GetRequest(s32 req_id, s32* out_err) {
    if (req_id == 0) {
        *out_err = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        *out_err = ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
        return nullptr;
    }
    auto& req = g_requests[req_index];
    if (req.state == NpRequestState::Complete) {
        req.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        *out_err = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (req.state == NpRequestState::Aborted) {
        req.result = ORBIS_NP_ERROR_ABORTED;
        *out_err = ORBIS_NP_ERROR_ABORTED;
        return nullptr;
    }
    return &req;
}

// Complete a request and return OK (or SIGNED_OUT for async when disabled).
static s32 CompleteRequest(NpRequest& req, s32 result) {
    req.state = NpRequestState::Complete;
    req.result = result;
    if (result != ORBIS_OK && req.async) {
        // Async requests always return OK immediately; result is read via PollAsync/WaitAsync.
        return ORBIS_OK;
    }
    return result;
}

s32 PS4_SYSV_ABI sceNpCreateRequest() {
    LOG_DEBUG(Lib_NpManager, "called");
    return CreateNpRequest(false);
}

s32 PS4_SYSV_ABI sceNpCreateAsyncRequest(const OrbisNpCreateAsyncRequestParameter* param) {
    LOG_DEBUG(Lib_NpManager, "called");
    if (param == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    if (param->size != sizeof(OrbisNpCreateAsyncRequestParameter)) {
        return ORBIS_NP_ERROR_INVALID_SIZE;
    }

    return CreateNpRequest(true);
}

s32 PS4_SYSV_ABI sceNpCheckNpAvailability(s32 req_id, OrbisNpOnlineId* online_id) {
    if (online_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    for (const User* user : UserManagement.GetLoggedInUsers()) {
        if (user == nullptr) {
            continue;
        }
        std::string helper_online_id;
        if (!GetLbp3HelperOnlineId(user->user_id, &helper_online_id)) {
            continue;
        }
        OrbisNpOnlineId helper_id{};
        SetNpOnlineId(helper_id, helper_online_id);
        if (std::memcmp(helper_id.data, online_id->data, sizeof(helper_id.data)) != 0) {
            continue;
        }
        std::scoped_lock lk{g_request_mutex};
        s32 err;
        NpRequest* req = GetRequest(req_id, &err);
        if (req == nullptr) {
            return err;
        }
        LOG_INFO(Lib_NpManager, "sceNpCheckNpAvailability: LBP3 helper identity '{}' is available",
                 helper_online_id);
        return CompleteRequest(*req, ORBIS_OK);
    }

    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    return sceNpCheckNpAvailabilityA(req_id, user_id);
}

s32 PS4_SYSV_ABI sceNpCheckNpAvailabilityA(s32 req_id,
                                           Libraries::UserService::OrbisUserServiceUserId user_id) {
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        LOG_ERROR(Lib_NpManager, "invalid user_id {}", user_id);
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    const bool helper_available = IsLbp3HelperAvailable(user_id);
    std::scoped_lock lk{g_request_mutex};
    s32 err;
    NpRequest* req = GetRequest(req_id, &err);
    if (!req)
        return err;
    if (helper_available) {
        return CompleteRequest(*req, ORBIS_OK);
    }
    if (!IsNpUserOnline(user_id)) {
        return CompleteRequest(*req, ORBIS_NP_ERROR_SIGNED_OUT);
    }
    LOG_DEBUG(Lib_NpManager, "req_id = {:#x}, user_id = {}", req_id, user_id);
    return CompleteRequest(*req, ORBIS_OK);
}

s32 PS4_SYSV_ABI sceNpCheckNpReachability(s32 req_id,
                                          Libraries::UserService::OrbisUserServiceUserId user_id) {
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    const bool helper_available = IsLbp3HelperAvailable(user_id);
    std::scoped_lock lk{g_request_mutex};
    s32 err;
    NpRequest* req = GetRequest(req_id, &err);
    if (!req)
        return err;
    if (helper_available) {
        return CompleteRequest(*req, ORBIS_OK);
    }
    if (!g_shadnet_enabled || !Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id)) {
        return CompleteRequest(*req, ORBIS_NP_ERROR_SIGNED_OUT);
    }
    LOG_DEBUG(Lib_NpManager, "req_id = {:#x}, user_id = {}", req_id, user_id);
    return CompleteRequest(*req, ORBIS_OK);
}

s32 PS4_SYSV_ABI sceNpCheckPlus(s32 req_id, const OrbisNpCheckPlusParameter* param,
                                OrbisNpCheckPlusResult* result) {
    if (req_id == 0 || param == nullptr || result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpCheckPlusParameter)) {
        return ORBIS_NP_ERROR_INVALID_SIZE;
    }
    if (param->user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (param->features < 1 || param->features > 3) {
        // TODO: If compiled SDK version is greater or equal to fw 3.50,
        // // error if param->features != 1 instead.
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // The reserved field must be zero-initialized by the caller.
    for (u8 b : param->reserved) {
        if (b != 0) {
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
    }
    std::scoped_lock lk{g_request_mutex};
    s32 err;
    NpRequest* req = GetRequest(req_id, &err);
    if (!req)
        return err;
    if (!IsNpUserOnline(param->user_id)) {
        return CompleteRequest(*req, ORBIS_NP_ERROR_SIGNED_OUT);
    }
    LOG_DEBUG(Lib_NpManager, "req_id = {:#x}, features = {:#x}", req_id, param->features);
    // Grant PS+: shadNet has no subscription gating.
    result->authorized = true;
    return CompleteRequest(*req, ORBIS_OK);
}

s32 PS4_SYSV_ABI sceNpGetAccountLanguage(s32 req_id, OrbisNpOnlineId* online_id,
                                         OrbisNpLanguageCode* language) {
    if (online_id == nullptr || language == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    return sceNpGetAccountLanguageA(req_id, user_id, language);
}

s32 PS4_SYSV_ABI sceNpGetAccountLanguageA(s32 req_id,
                                          Libraries::UserService::OrbisUserServiceUserId user_id,
                                          OrbisNpLanguageCode* language) {
    if (language == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    std::scoped_lock lk{g_request_mutex};
    s32 err;
    NpRequest* req = GetRequest(req_id, &err);
    if (!req)
        return err;
    if (!IsNpUserOnline(user_id)) {
        return CompleteRequest(*req, ORBIS_NP_ERROR_SIGNED_OUT);
    }
    LOG_DEBUG(Lib_NpManager, "req_id = {:#x}, user_id = {}", req_id, user_id);
    FillLanguageCodeFromProfile(user_id, language);
    return CompleteRequest(*req, ORBIS_OK);
}

s32 PS4_SYSV_ABI sceNpGetParentalControlInfo(s32 req_id, OrbisNpOnlineId* online_id, s8* age,
                                             OrbisNpParentalControlInfo* info) {
    if (online_id == nullptr || age == nullptr || info == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    return sceNpGetParentalControlInfoA(req_id, user_id, age, info);
}

s32 PS4_SYSV_ABI
sceNpGetParentalControlInfoA(s32 req_id, Libraries::UserService::OrbisUserServiceUserId user_id,
                             s8* age, OrbisNpParentalControlInfo* info) {
    if (age == nullptr || info == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    std::scoped_lock lk{g_request_mutex};
    s32 err;
    NpRequest* req = GetRequest(req_id, &err);
    if (!req)
        return err;
    if (!IsNpUserOnline(user_id)) {
        return CompleteRequest(*req, ORBIS_NP_ERROR_SIGNED_OUT);
    }
    LOG_DEBUG(Lib_NpManager, "req_id = {:#x}, user_id = {}", req_id, user_id);
    *age = GetAgeFromProfile(user_id);
    std::memset(info, 0, sizeof(OrbisNpParentalControlInfo));
    return CompleteRequest(*req, ORBIS_OK);
}

s32 PS4_SYSV_ABI sceNpAbortRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpManager, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (g_requests[req_index].state == NpRequestState::Complete) {
        // If the request is already complete, abort is ignored.
        return ORBIS_OK;
    }

    g_requests[req_index].state = NpRequestState::Aborted;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSetTimeout(s32 req_id, s32 resolve_retry, u32 resolve_timeout,
                                 u32 conn_timeout, u32 send_timeout, u32 recv_timeout) {
    LOG_DEBUG(Lib_NpManager, "called req_id = {:#x}", req_id);

    if (req_id <= 0) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    // ORBIS_NP_TIMEOUT_NO_EFFECT (0) is allowed per field (use default) but not for every field.
    const auto unset_or_at_least = [](u32 v, u32 min_us) { return v == 0 || v >= min_us; };
    const bool all_zero = resolve_retry == 0 && resolve_timeout == 0 && conn_timeout == 0 &&
                          send_timeout == 0 && recv_timeout == 0;
    if (all_zero || resolve_retry < 0 || !unset_or_at_least(resolve_timeout, 1'000'000) ||
        !unset_or_at_least(conn_timeout, 10'000'000) ||
        !unset_or_at_least(send_timeout, 10'000'000) ||
        !unset_or_at_least(recv_timeout, 10'000'000)) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    // shadNet performs no real network I/O, so the timeouts are validated and accepted but
    // not applied.
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWaitAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_requests[req_index].async || g_requests[req_index].state == NpRequestState::Ready) {
        return ORBIS_NP_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_requests[req_index].result;
    LOG_WARNING(Lib_NpManager, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpPollAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_requests[req_index].async || g_requests[req_index].state == NpRequestState::Ready) {
        return ORBIS_NP_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_requests[req_index].result;
    LOG_WARNING(Lib_NpManager, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpDeleteRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpManager, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    g_active_requests--;
    g_requests[req_index].state = NpRequestState::None;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountCountry(OrbisNpOnlineId* online_id,
                                        OrbisNpCountryCode* country_code) {
    if (online_id == nullptr || country_code == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (!IsNpUserOnline(user_id)) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    FillCountryCodeFromProfile(user_id, country_code);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountCountryA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                         OrbisNpCountryCode* country_code) {
    if (country_code == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (UserManagement.GetUserByID(user_id) == nullptr) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (!IsNpUserOnline(user_id)) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    FillCountryCodeFromProfile(user_id, country_code);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountDateOfBirth(OrbisNpOnlineId* online_id,
                                            OrbisNpDate* date_of_birth) {
    if (online_id == nullptr || date_of_birth == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (!IsNpUserOnline(user_id)) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    FillDateOfBirthFromProfile(user_id, date_of_birth);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountDateOfBirthA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpDate* date_of_birth) {
    if (date_of_birth == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (UserManagement.GetUserByID(user_id) == nullptr) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (!IsNpUserOnline(user_id)) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    FillDateOfBirthFromProfile(user_id, date_of_birth);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetGamePresenceStatus(OrbisNpOnlineId* online_id,
                                            OrbisNpGamePresenseStatus* game_status) {
    if (online_id == nullptr || game_status == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    *game_status = user_id != -1 && IsNpUserOnline(user_id) ? OrbisNpGamePresenseStatus::Online
                                                            : OrbisNpGamePresenseStatus::Offline;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetGamePresenceStatusA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpGamePresenseStatus* game_status) {
    if (game_status == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    *game_status = IsNpUserOnline(user_id) ? OrbisNpGamePresenseStatus::Online
                                           : OrbisNpGamePresenseStatus::Offline;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountId(OrbisNpOnlineId* online_id, u64* account_id) {
    LOG_DEBUG(Lib_NpManager, "called");
    if (online_id == nullptr || account_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    s32 user_id = FindLbp3HelperUserByOnlineId(*online_id);
    if (user_id == -1) {
        user_id = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    }
    if (user_id == -1) {
        *account_id = 0;
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (!IsNpUserOnline(user_id)) {
        *account_id = 0;
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    *account_id = Lbp3HelperAccountId(user_id);
    if (*account_id == 0) {
        *account_id = Libraries::Np::NpHandler::GetInstance().GetAccountId(user_id);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountIdA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                    u64* account_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (account_id == nullptr ||
        user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!IsNpUserOnline(user_id)) {
        *account_id = 0;
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    *account_id = Lbp3HelperAccountId(user_id);
    if (*account_id == 0) {
        *account_id = Libraries::Np::NpHandler::GetInstance().GetAccountId(user_id);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetNpId(Libraries::UserService::OrbisUserServiceUserId user_id,
                              OrbisNpId* np_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return (g_firmware_version >= 0 && g_firmware_version < Common::ElfInfo::FW_900)
                   ? ORBIS_NP_ERROR_USER_NOT_FOUND
                   : ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (np_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    std::string helper_online_id;
    if (GetLbp3HelperOnlineId(user_id, &helper_online_id)) {
        SetNpId(*np_id, helper_online_id);
        LOG_INFO(Lib_NpManager, "sceNpGetNpId: LBP3 helper identity user_id={} OnlineID='{}'",
                 user_id, helper_online_id);
        return ORBIS_OK;
    }
    if (!g_shadnet_enabled || !Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id)) {
        LOG_WARNING(Lib_NpManager,
                    "sceNpGetNpId: no network NP account (user_id={} shadnet_enabled={} "
                    "signed_in={})",
                    user_id, g_shadnet_enabled,
                    Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id));
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    *np_id = Libraries::Np::NpHandler::GetInstance().GetNpId(user_id);
    LOG_INFO(Lib_NpManager, "sceNpGetNpId: user_id={} handle.data='{}' (strnlen={})", user_id,
             np_id->handle.data, strnlen(np_id->handle.data, sizeof(np_id->handle.data)));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetOnlineId(Libraries::UserService::OrbisUserServiceUserId user_id,
                                  OrbisNpOnlineId* online_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return (g_firmware_version >= 0 && g_firmware_version < Common::ElfInfo::FW_900)
                   ? ORBIS_NP_ERROR_USER_NOT_FOUND
                   : ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (online_id == nullptr) {
        LOG_ERROR(Lib_NpManager, "invalid argument: online_id is null");
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    std::string helper_online_id;
    if (GetLbp3HelperOnlineId(user_id, &helper_online_id)) {
        SetNpOnlineId(*online_id, helper_online_id);
        LOG_INFO(Lib_NpManager, "sceNpGetOnlineId: LBP3 helper identity user_id={} OnlineID='{}'",
                 user_id, helper_online_id);
        return ORBIS_OK;
    }
    if (!g_shadnet_enabled || !Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id)) {
        LOG_INFO(Lib_NpManager,
                 "sceNpGetOnlineId: SIGNED_OUT (user_id={} shadnet_enabled={} signed_in={})",
                 user_id, g_shadnet_enabled,
                 Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id));
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    *online_id = Libraries::Np::NpHandler::GetInstance().GetOnlineId(user_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetNpReachabilityState(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpReachabilityState* state) {
    if (state == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        if (g_firmware_version < 0 || g_firmware_version >= Common::ElfInfo::FW_400) {
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
    }
    const bool reachable =
        IsLbp3HelperAvailable(user_id) ||
        (g_shadnet_enabled && Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id));
    *state =
        reachable ? OrbisNpReachabilityState::Reachable : OrbisNpReachabilityState::Unavailable;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetState(Libraries::UserService::OrbisUserServiceUserId user_id,
                               OrbisNpState* state) {
    if (state == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        if (g_firmware_version < 0 || g_firmware_version >= Common::ElfInfo::FW_900) {
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
    }
    if (IsLbp3HelperAvailable(user_id)) {
        *state = OrbisNpState::SignedIn;
        LOG_DEBUG(Lib_NpManager, "LBP3 helper connected, local NP state=SignedIn");
        return ORBIS_OK;
    }
    if (!g_shadnet_enabled) {
        *state = OrbisNpState::SignedOut;
        LOG_DEBUG(Lib_NpManager, "shadNet disabled,SignedOut");
        return ORBIS_OK;
    }
    *state = Libraries::Np::NpHandler::GetInstance().IsPsnSignedIn(user_id)
                 ? OrbisNpState::SignedIn
                 : OrbisNpState::SignedOut;
    LOG_DEBUG(Lib_NpManager, "user_id={} state={}", user_id,
              *state == OrbisNpState::SignedIn ? "SignedIn" : "SignedOut");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI
sceNpGetUserIdByAccountId(u64 account_id, Libraries::UserService::OrbisUserServiceUserId* user_id) {
    if (account_id == 0 || user_id == nullptr) {
        LOG_ERROR(Lib_NpManager, "invalid argument: account_id={}", account_id);
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    for (const User* user : UserManagement.GetLoggedInUsers()) {
        if (user != nullptr && Lbp3HelperAccountId(user->user_id) == account_id) {
            *user_id = user->user_id;
            return ORBIS_OK;
        }
    }
    if (!g_shadnet_enabled)
        return ORBIS_NP_ERROR_SIGNED_OUT;

    const s32 found = Libraries::Np::NpHandler::GetInstance().GetUserIdByAccountId(account_id);
    if (found == -1)
        return ORBIS_NP_ERROR_SIGNED_OUT;

    // Verify the resolved user_id is actually a logged-in local user
    const User* u = UserManagement.GetUserByID(found);
    if (!u || !u->logged_in)
        return ORBIS_NP_ERROR_USER_NOT_FOUND;

    *user_id = found;
    LOG_DEBUG(Lib_NpManager, "account_id={} returns user_id={}", account_id, *user_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetUserIdByOnlineId(const OrbisNpOnlineId* online_id,
                                          Libraries::UserService::OrbisUserServiceUserId* user_id) {
    if (online_id == nullptr || user_id == nullptr) {
        LOG_ERROR(Lib_NpManager, "invalid argument");
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (const s32 helper_user = FindLbp3HelperUserByOnlineId(*online_id); helper_user != -1) {
        *user_id = helper_user;
        return ORBIS_OK;
    }
    if (!g_shadnet_enabled)
        return ORBIS_NP_ERROR_SIGNED_OUT;

    const s32 found = Libraries::Np::NpHandler::GetInstance().GetUserIdByOnlineId(*online_id);
    if (found == -1)
        return ORBIS_NP_ERROR_SIGNED_OUT;

    // Verify the resolved user_id is actually a logged-in local user
    const User* u = UserManagement.GetUserByID(found);
    if (!u || !u->logged_in)
        return ORBIS_NP_ERROR_USER_NOT_FOUND;

    *user_id = found;
    LOG_DEBUG(Lib_NpManager, "returns user_id={}", *user_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpHasSignedUp(Libraries::UserService::OrbisUserServiceUserId user_id,
                                  bool* has_signed_up) {
    LOG_DEBUG(Lib_NpManager, "called");
    if (has_signed_up == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    *has_signed_up = false;
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        if (g_firmware_version < 0 || g_firmware_version >= Common::ElfInfo::FW_900) {
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
    }
    const User* u = UserManagement.GetUserByID(user_id);
    if (u == nullptr) {
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }
    if (IsLbp3HelperAvailable(user_id)) {
        *has_signed_up = true;
        return ORBIS_OK;
    }
    // A user has signed up if they have a shadNet npid configured.
    // This is independent of shadnet_enabled and current connection state.
    *has_signed_up = !u->shadnet_npid.empty();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSetContentRestriction(const OrbisNpContentRestriction* restriction) {
    LOG_ERROR(Lib_NpManager, "(STUBBED) called");
    if (restriction == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (restriction->size != sizeof(OrbisNpContentRestriction)) {
        return ORBIS_NP_ERROR_INVALID_SIZE;
    }
    if (restriction->default_age_restriction < 0 || restriction->age_restriction_count > 0x100) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (restriction->age_restriction_count > 0 && restriction->age_restriction == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSetNpTitleId(const OrbisNpTitleId* title_id,
                                   const OrbisNpTitleSecret* title_secret) {
    if (title_id == nullptr || title_secret == nullptr) {
        LOG_ERROR(Lib_NpManager, "called with invalid arguments");
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    LOG_ERROR(Lib_NpManager, "(STUBBED) called, title_id = {}", title_id->id);
    return ORBIS_OK;
}

struct NpStateCallbackForNpToolkit {
    OrbisNpStateCallbackForNpToolkit func;
    void* userdata;
    std::map<Libraries::UserService::OrbisUserServiceUserId, u64> last_sequence;
};

NpStateCallbackForNpToolkit NpStateCbForNp;

struct LegacyNpStateCallback {
    OrbisNpStateCallback func;
    void* userdata;
    std::map<Libraries::UserService::OrbisUserServiceUserId, u64> last_sequence;
};

LegacyNpStateCallback LegacyNpStateCb;

struct NpStateCallbackAEntry {
    OrbisNpStateCallbackA func;
    void* userdata;
    bool in_use;
    std::map<Libraries::UserService::OrbisUserServiceUserId, u64> last_sequence;
};

static std::array<NpStateCallbackAEntry, ORBIS_NP_STATE_CALLBACK_MAX> g_np_state_callbacks{};

struct NpReachabilityStateCallback {
    OrbisNpReachabilityStateCallback func;
    void* userdata;
};

NpReachabilityStateCallback NpReachabilityCb;

// Last reachability state delivered to NpReachabilityCb, per user. Keeps the reachability
// callback edge-triggered: it fires only when the derived state changes for that user.
static std::map<Libraries::UserService::OrbisUserServiceUserId, OrbisNpReachabilityState>
    g_np_reachability_last;

struct PendingNpStateEvent {
    Libraries::UserService::OrbisUserServiceUserId user_id;
    OrbisNpState state;
    OrbisNpId np_id;
    bool has_np_id;
    u64 sequence;
};

static std::deque<PendingNpStateEvent> g_np_state_events;
static std::map<Libraries::UserService::OrbisUserServiceUserId, PendingNpStateEvent>
    g_np_sticky_states;
static u64 g_np_state_sequence{};
static bool g_lbp3_legacy_signed_in_deferred_logged{};

// CUSA00063 v1.26 records the first state delivered to its legacy NP callback before checking
// whether its online-presentation owner is ready. A SignedIn delivered immediately after
// sceNpRegisterStateCallback is therefore remembered as state 2 but cannot create the EULA task;
// every later sticky replay is discarded by the title as a duplicate.
//
// The local helper can already be connected before the callback is registered, so there is no
// preceding SignedOut edge. By the time the presentation owner reaches state 3, its transition
// latch at +0x338 has been asserted. The title's SignedOut branch normally clears that latch, and
// its SignedIn branch refuses to run while it remains set. Hold this title's sticky SignedIn until
// the owner is ready. The stale pre-registration latch is hidden only for the duration of the
// callback below; it is also the online manager's success flag and must remain asserted otherwise.
// Other registered callbacks and reachability notifications retain their normal timing.
static bool IsLbp3LegacySignedInReady(const LegacyNpStateCallback& callback) {
    if (!Net::Lbp3OnlineBridge::IsSupportedTitle() || MemoryPatcher::g_eboot_address == 0 ||
        callback.func == nullptr) {
        return true;
    }

    constexpr uintptr_t LegacyCallbackOffset = 0x0033f1e0;
    constexpr uintptr_t LegacyCallbackUserdataOffset = 0x0130e248;
    constexpr uintptr_t PresentationOwnerStateOffset = 0x0130d9a4; // owner + 0x2ec
    constexpr uintptr_t CallbackReadyFlagOffset = 0x0130e409;      // userdata + 0x1c1

    const uintptr_t base = MemoryPatcher::g_eboot_address;
    if (reinterpret_cast<uintptr_t>(callback.func) != base + LegacyCallbackOffset ||
        reinterpret_cast<uintptr_t>(callback.userdata) != base + LegacyCallbackUserdataOffset) {
        return true;
    }

    const auto* owner_state =
        reinterpret_cast<const volatile u32*>(base + PresentationOwnerStateOffset);
    const auto* callback_ready =
        reinterpret_cast<const volatile u8*>(base + CallbackReadyFlagOffset);
    return *owner_state == 3 && *callback_ready != 0;
}

static volatile u8* GetLbp3LegacyPresentationTransitionLatch(
    const LegacyNpStateCallback& callback) {
    if (!Net::Lbp3OnlineBridge::IsSupportedTitle() || MemoryPatcher::g_eboot_address == 0 ||
        callback.func == nullptr) {
        return nullptr;
    }

    constexpr uintptr_t LegacyCallbackOffset = 0x0033f1e0;
    constexpr uintptr_t LegacyCallbackUserdataOffset = 0x0130e248;
    constexpr uintptr_t PresentationTransitionLatchOffset = 0x0130d9f0;
    const uintptr_t base = MemoryPatcher::g_eboot_address;
    if (reinterpret_cast<uintptr_t>(callback.func) != base + LegacyCallbackOffset ||
        reinterpret_cast<uintptr_t>(callback.userdata) != base + LegacyCallbackUserdataOffset) {
        return nullptr;
    }
    return reinterpret_cast<volatile u8*>(base + PresentationTransitionLatchOffset);
}

static void QueueNpStateEvent(Libraries::UserService::OrbisUserServiceUserId user_id,
                              OrbisNpState state) {
    PendingNpStateEvent event{};
    event.user_id = user_id;
    event.state = state;
    event.has_np_id = state == OrbisNpState::SignedIn;
    if (event.has_np_id) {
        std::string helper_online_id;
        if (GetLbp3HelperOnlineId(user_id, &helper_online_id)) {
            SetNpId(event.np_id, helper_online_id);
        } else {
            // NpId is built from the user's shadnet_npid at login; GetNpId returns by value.
            event.np_id = Libraries::Np::NpHandler::GetInstance().GetNpId(user_id);
        }
    }

    std::scoped_lock lk{g_np_state_events_mutex};
    event.sequence = ++g_np_state_sequence;
    g_np_sticky_states.insert_or_assign(user_id, event);
    g_np_state_events.emplace_back(event);
}

static void EnsureLbp3HelperSignedInSticky(const char* reason) {
    if (!Net::Lbp3OnlineBridge::IsSupportedTitle()) {
        return;
    }

    for (const User* user : UserManagement.GetLoggedInUsers()) {
        if (user == nullptr) {
            continue;
        }
        std::string helper_online_id;
        if (!GetLbp3HelperOnlineId(user->user_id, &helper_online_id)) {
            continue;
        }

        bool already_signed_in = false;
        {
            std::scoped_lock lk{g_np_state_events_mutex};
            const auto it = g_np_sticky_states.find(user->user_id);
            already_signed_in =
                it != g_np_sticky_states.end() && it->second.state == OrbisNpState::SignedIn;
        }
        if (already_signed_in) {
            continue;
        }

        QueueNpStateEvent(user->user_id, OrbisNpState::SignedIn);
        LOG_CRITICAL(Lib_NpManager,
                     "LBP3 helper forced sticky SignedIn after {}: user_id={} OnlineID='{}'",
                     reason, user->user_id, helper_online_id);
    }
}

static void LogStickySignedInReplay(const char* callback_kind) {
    EnsureLbp3HelperSignedInSticky(callback_kind);
    size_t signed_in_count = 0;
    {
        std::scoped_lock lk{g_np_state_events_mutex};
        for (const auto& [user_id, event] : g_np_sticky_states) {
            if (event.state == OrbisNpState::SignedIn) {
                ++signed_in_count;
            }
        }
    }
    if (signed_in_count != 0) {
        LOG_INFO(Lib_NpManager,
                 "NP {} callback registered late; armed sticky SignedIn replay for {} user(s)",
                 callback_kind, signed_in_count);
    } else {
        LOG_INFO(Lib_NpManager, "NP {} state callback registered", callback_kind);
    }
}

void NotifyNpStateFromUserServiceEvent(Libraries::UserService::OrbisUserServiceEventType event_type,
                                       Libraries::UserService::OrbisUserServiceUserId user_id) {
    switch (event_type) {
    case Libraries::UserService::OrbisUserServiceEventType::Login:
        if (IsLbp3HelperAvailable(user_id)) {
            QueueNpStateEvent(user_id, OrbisNpState::SignedIn);
            LOG_INFO(Lib_NpManager, "LBP3 helper queued initial SignedIn for user_id={}", user_id);
        } else if (g_shadnet_enabled) {
            // Handler connects the user and fires SignedIn via the bridge on success.
            Libraries::Np::NpHandler::GetInstance().OnUserLoggedIn(user_id);
        } else {
            // A local console login does not transition an already-offline NP user.
            LOG_INFO(Lib_NpManager,
                     "Suppressing redundant SignedOut event for offline local login, user_id={}",
                     user_id);
        }
        break;
    case Libraries::UserService::OrbisUserServiceEventType::Logout:
        if (Net::Lbp3OnlineBridge::IsSupportedTitle()) {
            QueueNpStateEvent(user_id, OrbisNpState::SignedOut);
        } else if (g_shadnet_enabled) {
            // Handler disconnects the user and fires SignedOut via the bridge.
            Libraries::Np::NpHandler::GetInstance().OnUserLoggedOut(user_id);
        } else {
            // The NP user was never signed in, so local logout is not an NP state edge.
            LOG_INFO(Lib_NpManager,
                     "Suppressing redundant SignedOut event for offline local logout, user_id={}",
                     user_id);
        }
        break;
    default:
        break;
    }
}

static s32 RegisterStateCallbackA(OrbisNpStateCallbackA callback, void* userdata) {
    if (callback == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_np_state_callbacks_mutex};

    for (const auto& entry : g_np_state_callbacks) {
        if (!entry.in_use) {
            continue;
        }
        if (entry.func == callback) {
            return ORBIS_NP_ERROR_CALLBACK_ALREADY_REGISTERED;
        }
    }

    for (size_t i = 0; i < g_np_state_callbacks.size(); ++i) {
        auto& entry = g_np_state_callbacks[i];
        if (entry.in_use) {
            continue;
        }
        entry.func = callback;
        entry.userdata = userdata;
        entry.in_use = true;
        entry.last_sequence.clear();
        return static_cast<s32>(i + 1);
    }

    return ORBIS_NP_ERROR_CALLBACK_MAX;
}

static s32 UnregisterStateCallbackAById(s32 callback_id) {
    if (callback_id <= 0 || callback_id > static_cast<s32>(g_np_state_callbacks.size())) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_np_state_callbacks_mutex};

    auto& entry = g_np_state_callbacks[callback_id - 1];
    if (!entry.in_use) {
        return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
    }

    entry = {};
    return ORBIS_OK;
}

static void DispatchPendingNpStateCallbacks() {
    struct PreparedDispatch {
        PendingNpStateEvent event;
        bool legacy{};
        std::array<bool, ORBIS_NP_STATE_CALLBACK_MAX> callback_a{};
        bool toolkit{};
        bool sticky_replay{};
    };

    std::deque<PendingNpStateEvent> candidate_events;
    std::vector<PreparedDispatch> dispatches;
    LegacyNpStateCallback legacy_callback{};
    NpStateCallbackForNpToolkit toolkit_callback{};
    NpReachabilityStateCallback reachability_callback{};
    std::array<NpStateCallbackAEntry, ORBIS_NP_STATE_CALLBACK_MAX> callbacks;
    std::vector<std::pair<Libraries::UserService::OrbisUserServiceUserId, OrbisNpReachabilityState>>
        reachability_changes;
    {
        std::scoped_lock lk{g_np_state_events_mutex, g_np_state_callbacks_mutex};
        if (g_np_state_events.empty() && g_np_sticky_states.empty()) {
            return;
        }
        candidate_events.swap(g_np_state_events);
        const size_t queued_event_count = candidate_events.size();
        for (const auto& [user_id, event] : g_np_sticky_states) {
            candidate_events.emplace_back(event);
        }

        legacy_callback = LegacyNpStateCb;
        callbacks = g_np_state_callbacks;
        toolkit_callback = NpStateCbForNp;
        reachability_callback = NpReachabilityCb;

        dispatches.reserve(candidate_events.size());
        for (size_t event_index = 0; event_index < candidate_events.size(); ++event_index) {
            const auto& event = candidate_events[event_index];
            PreparedDispatch dispatch{
                .event = event,
                .sticky_replay = event_index >= queued_event_count,
            };

            if (LegacyNpStateCb.func != nullptr &&
                LegacyNpStateCb.last_sequence[event.user_id] < event.sequence) {
                const bool defer_lbp3_signed_in = event.state == OrbisNpState::SignedIn &&
                                                  !IsLbp3LegacySignedInReady(LegacyNpStateCb);
                if (!defer_lbp3_signed_in) {
                    LegacyNpStateCb.last_sequence[event.user_id] = event.sequence;
                    dispatch.legacy = true;
                    if (event.state == OrbisNpState::SignedIn &&
                        g_lbp3_legacy_signed_in_deferred_logged) {
                        LOG_CRITICAL(Lib_NpManager,
                                     "LBP3 deferred legacy SignedIn is now UI-ready; "
                                     "dispatching sequence {}",
                                     event.sequence);
                        g_lbp3_legacy_signed_in_deferred_logged = false;
                    }
                } else if (!g_lbp3_legacy_signed_in_deferred_logged) {
                    LOG_CRITICAL(Lib_NpManager,
                                 "LBP3 legacy SignedIn deferred until the title's EULA/UI "
                                 "presentation owner is ready");
                    g_lbp3_legacy_signed_in_deferred_logged = true;
                }
            }

            for (size_t i = 0; i < g_np_state_callbacks.size(); ++i) {
                auto& entry = g_np_state_callbacks[i];
                if (entry.in_use && entry.func != nullptr &&
                    entry.last_sequence[event.user_id] < event.sequence) {
                    entry.last_sequence[event.user_id] = event.sequence;
                    dispatch.callback_a[i] = true;
                }
            }

            if (NpStateCbForNp.func != nullptr &&
                NpStateCbForNp.last_sequence[event.user_id] < event.sequence) {
                NpStateCbForNp.last_sequence[event.user_id] = event.sequence;
                dispatch.toolkit = true;
            }

            if (reachability_callback.func != nullptr) {
                const OrbisNpReachabilityState reach = event.state == OrbisNpState::SignedIn
                                                           ? OrbisNpReachabilityState::Reachable
                                                           : OrbisNpReachabilityState::Unavailable;
                auto it = g_np_reachability_last.find(event.user_id);
                if (it == g_np_reachability_last.end() || it->second != reach) {
                    g_np_reachability_last[event.user_id] = reach;
                    reachability_changes.emplace_back(event.user_id, reach);
                }
            }

            if (dispatch.legacy || dispatch.toolkit ||
                std::ranges::any_of(dispatch.callback_a, [](bool enabled) { return enabled; })) {
                dispatches.emplace_back(std::move(dispatch));
            }
        }
    }

    for (auto& dispatch : dispatches) {
        auto& event = dispatch.event;
        const size_t callback_a_count = std::ranges::count(dispatch.callback_a, true);
        if (event.state == OrbisNpState::SignedIn) {
            LOG_CRITICAL(Lib_NpManager,
                         "Dispatching {}SignedIn seq={} for user_id={}: legacy={}, callback_a={}, "
                         "toolkit={}",
                         dispatch.sticky_replay ? "sticky " : "", event.sequence, event.user_id,
                         dispatch.legacy, callback_a_count, dispatch.toolkit);
        }

        if (dispatch.legacy) {
            volatile u8* transition_latch = nullptr;
            u8 saved_transition_latch = 0;
            bool transition_latch_masked = false;
            if (event.state == OrbisNpState::SignedIn) {
                transition_latch = GetLbp3LegacyPresentationTransitionLatch(legacy_callback);
                if (transition_latch != nullptr) {
                    saved_transition_latch = *transition_latch;
                    if (saved_transition_latch != 0) {
                        *transition_latch = 0;
                        transition_latch_masked = true;
                    }
                }
            }
            legacy_callback.func(event.user_id, event.state,
                                 event.has_np_id ? &event.np_id : nullptr,
                                 legacy_callback.userdata);
            if (transition_latch_masked) {
                *transition_latch = saved_transition_latch;
            }
        }

        for (size_t i = 0; i < callbacks.size(); ++i) {
            if (dispatch.callback_a[i]) {
                callbacks[i].func(event.user_id, event.state, callbacks[i].userdata);
            }
        }

        if (dispatch.toolkit) {
            toolkit_callback.func(event.user_id, event.state, toolkit_callback.userdata);
        }
    }

    // Reachability callback fires only on a change, after the state callbacks.
    for (const auto& [user_id, reach] : reachability_changes) {
        reachability_callback.func(user_id, reach, reachability_callback.userdata);
    }
}

s32 PS4_SYSV_ABI sceNpCheckCallback() {
    LOG_DEBUG(Lib_NpManager, "called");
    EnsureLbp3HelperSignedInSticky("sceNpCheckCallback");
    DispatchPendingNpStateCallbacks();
    Net::Lbp3OnlineBridge::MaybeQueueFindBestRoom();

    std::scoped_lock lk{g_np_callbacks_mutex};
    for (auto& [key, cb] : g_np_callbacks) {
        cb();
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpCheckCallbackForLib() {
    LOG_DEBUG(Lib_NpManager, "(STUBBED) called");
    EnsureLbp3HelperSignedInSticky("sceNpCheckCallbackForLib");
    DispatchPendingNpStateCallbacks();
    Net::Lbp3OnlineBridge::MaybeQueueFindBestRoom();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpRegisterStateCallback(OrbisNpStateCallback callback, void* userdata) {
    if (callback == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    {
        std::scoped_lock lk{g_np_state_callbacks_mutex};
        if (LegacyNpStateCb.func != nullptr) {
            return ORBIS_NP_ERROR_CALLBACK_ALREADY_REGISTERED;
        }

        LOG_CRITICAL(Lib_NpManager,
                     "LBP3 NP legacy state callback registered: callback={:#x}, guest_va={:#x}, "
                     "userdata={}",
                     reinterpret_cast<uintptr_t>(callback),
                     reinterpret_cast<uintptr_t>(callback) - MemoryPatcher::g_eboot_address,
                     userdata);
        LegacyNpStateCb.func = callback;
        LegacyNpStateCb.userdata = userdata;
        LegacyNpStateCb.last_sequence.clear();
    }

    LogStickySignedInReplay("legacy");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUnregisterStateCallback() {
    std::scoped_lock lk{g_np_state_callbacks_mutex};
    if (LegacyNpStateCb.func == nullptr) {
        return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
    }
    LegacyNpStateCb.func = nullptr;
    LegacyNpStateCb.userdata = nullptr;
    LegacyNpStateCb.last_sequence.clear();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpRegisterStateCallbackA(OrbisNpStateCallbackA callback, void* userdata) {
    LOG_CRITICAL(Lib_NpManager,
                 "LBP3 NP A state callback registered: callback={:#x}, guest_va={:#x}, userdata={}",
                 reinterpret_cast<uintptr_t>(callback),
                 reinterpret_cast<uintptr_t>(callback) - MemoryPatcher::g_eboot_address, userdata);
    const s32 callback_id = RegisterStateCallbackA(callback, userdata);
    if (callback_id > 0) {
        LogStickySignedInReplay("A");
    }
    return callback_id;
}

s32 PS4_SYSV_ABI sceNpUnregisterStateCallbackA(s32 callback_id) {
    LOG_INFO(Lib_NpManager, "called, callback_id = {}", callback_id);
    return UnregisterStateCallbackAById(callback_id);
}

s32 PS4_SYSV_ABI sceNpRegisterNpReachabilityStateCallback(OrbisNpReachabilityStateCallback callback,
                                                          void* userdata) {
    if (callback == nullptr) {
        LOG_ERROR(Lib_NpManager, "callback is nullptr");
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    {
        std::scoped_lock lk{g_np_state_callbacks_mutex};
        if (NpReachabilityCb.func != nullptr) {
            LOG_ERROR(Lib_NpManager, "callback already registered, cannot register multiple");
            return ORBIS_NP_ERROR_CALLBACK_ALREADY_REGISTERED;
        }
        LOG_INFO(Lib_NpManager, "called");
        NpReachabilityCb.func = callback;
        NpReachabilityCb.userdata = userdata;
        // Reset the per-user cache so the next state transition reports fresh.
        g_np_reachability_last.clear();
    }
    EnsureLbp3HelperSignedInSticky("reachability callback registration");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUnregisterNpReachabilityStateCallback() {
    std::scoped_lock lk{g_np_state_callbacks_mutex};
    if (NpReachabilityCb.func == nullptr) {
        LOG_ERROR(Lib_NpManager, "callback not registered");
        return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
    }
    NpReachabilityCb.func = nullptr;
    NpReachabilityCb.userdata = nullptr;
    g_np_reachability_last.clear();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpRegisterStateCallbackForToolkit(OrbisNpStateCallbackForNpToolkit callback,
                                                      void* userdata) {
    LOG_CRITICAL(
        Lib_NpManager,
        "LBP3 NP toolkit state callback registered: callback={:#x}, guest_va={:#x}, userdata={}",
        reinterpret_cast<uintptr_t>(callback),
        reinterpret_cast<uintptr_t>(callback) - MemoryPatcher::g_eboot_address, userdata);
    {
        std::scoped_lock lk{g_np_state_callbacks_mutex};
        NpStateCbForNp.func = callback;
        NpStateCbForNp.userdata = userdata;
        NpStateCbForNp.last_sequence.clear();
    }
    LogStickySignedInReplay("toolkit");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUnregisterStateCallbackForToolkit() {
    std::scoped_lock lk{g_np_state_callbacks_mutex};
    if (NpStateCbForNp.func == nullptr) {
        return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
    }
    NpStateCbForNp.func = nullptr;
    NpStateCbForNp.userdata = nullptr;
    NpStateCbForNp.last_sequence.clear();
    return ORBIS_OK;
}

// shadNet has no live game-presence or PS Plus event source, so registered callbacks are
// stored but never invoked
using OrbisNpGamePresenceCallback = PS4_SYSV_ABI void (*)(const OrbisNpOnlineId* online_id,
                                                          void* userdata);
using OrbisNpPlusEventCallback = PS4_SYSV_ABI void (*)(
    Libraries::UserService::OrbisUserServiceUserId user_id, s32 event, void* userdata);

static std::mutex g_presence_plus_mutex;

static OrbisNpGamePresenceCallback g_game_presence_cb = nullptr;
static void* g_game_presence_cb_userdata = nullptr;

struct GamePresenceCallbackEntry {
    s32 handle;
    OrbisNpGamePresenceCallback func;
    void* userdata;
    bool in_use;
};
static std::array<GamePresenceCallbackEntry, ORBIS_NP_STATE_CALLBACK_MAX> g_game_presence_cbs_a{};
static s32 g_game_presence_next_handle = 0;

static OrbisNpPlusEventCallback g_plus_event_cb = nullptr;
static void* g_plus_event_cb_userdata = nullptr;

s32 PS4_SYSV_ABI sceNpSetGamePresenceOnline(s32 req_id, const OrbisNpOnlineId* online_id,
                                            u32 presence) {
    if (req_id == 0 || online_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};
    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || req_index < 0 ||
        req_index >= static_cast<s32>(g_requests.size()) ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    // No live presence service under shadNet; accept and report success.
    LOG_DEBUG(Lib_NpManager, "(STUBBED) called req_id = {:#x}", req_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSetGamePresenceOnlineA(s32 req_id,
                                             Libraries::UserService::OrbisUserServiceUserId user_id,
                                             u32 presence) {
    if (user_id == Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    const OrbisNpOnlineId online_id = Libraries::Np::NpHandler::GetInstance().GetOnlineId(user_id);
    return sceNpSetGamePresenceOnline(req_id, &online_id, presence);
}

void PS4_SYSV_ABI sceNpRegisterGamePresenceCallback(OrbisNpGamePresenceCallback callback,
                                                    void* userdata) {
    std::scoped_lock lk{g_presence_plus_mutex};
    g_game_presence_cb = callback;
    g_game_presence_cb_userdata = userdata;
}

s32 PS4_SYSV_ABI sceNpRegisterGamePresenceCallbackA(OrbisNpGamePresenceCallback callback,
                                                    void* userdata) {
    if (callback == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_presence_plus_mutex};
    for (auto& entry : g_game_presence_cbs_a) {
        if (entry.in_use) {
            continue;
        }
        entry.handle = ++g_game_presence_next_handle;
        entry.func = callback;
        entry.userdata = userdata;
        entry.in_use = true;
        return entry.handle;
    }
    return ORBIS_NP_ERROR_CALLBACK_MAX;
}

s32 PS4_SYSV_ABI sceNpUnregisterGamePresenceCallbackA(s32 callback_id) {
    std::scoped_lock lk{g_presence_plus_mutex};
    for (auto& entry : g_game_presence_cbs_a) {
        if (entry.in_use && entry.handle == callback_id) {
            entry = {};
            return ORBIS_OK;
        }
    }
    return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
}

s32 PS4_SYSV_ABI sceNpIsPlusMember(Libraries::UserService::OrbisUserServiceUserId user_id,
                                   bool* result) {
    if (result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // shadNet has no PS Plus concept; report not a Plus member.
    *result = false;
    LOG_DEBUG(Lib_NpManager, "user_id {} not a Plus member", user_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpRegisterPlusEventCallback(OrbisNpPlusEventCallback callback, void* userdata) {
    if (callback == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    std::scoped_lock lk{g_presence_plus_mutex};
    if (g_plus_event_cb != nullptr) {
        return ORBIS_NP_ERROR_CALLBACK_ALREADY_REGISTERED;
    }
    g_plus_event_cb = callback;
    g_plus_event_cb_userdata = userdata;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUnregisterPlusEventCallback() {
    std::scoped_lock lk{g_presence_plus_mutex};
    if (g_plus_event_cb == nullptr) {
        return ORBIS_NP_ERROR_CALLBACK_NOT_REGISTERED;
    }
    g_plus_event_cb = nullptr;
    g_plus_event_cb_userdata = nullptr;
    return ORBIS_OK;
}

void RegisterNpCallback(std::string key, std::function<void()> cb) {
    std::scoped_lock lk{g_np_callbacks_mutex};
    LOG_DEBUG(Lib_NpManager, "registering callback processing for {}", key);
    g_np_callbacks.emplace(key, cb);
}

void DeregisterNpCallback(std::string key) {
    std::scoped_lock lk{g_np_callbacks_mutex};
    LOG_DEBUG(Lib_NpManager, "deregistering callback processing for {}", key);
    g_np_callbacks.erase(key);
}

// The bandwidth-test service is unavailable while shadNet is disabled, but callers still
// expect its asynchronous state machine to reach a terminal state. A zero-returning common
// stub starts a test with handle 0 and then leaves the status output untouched forever.
static s32 PS4_SYSV_ABI sceNpBandwidthTestInitStart(const void* param) {
    if (param == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    LOG_DEBUG(Lib_NpManager, "completing bandwidth test offline");
    return 1;
}

static s32 PS4_SYSV_ABI sceNpBandwidthTestGetStatus(s32 test_id, s32* status) {
    if (test_id < 0 || status == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // LBP3 treats status 2 as complete and then calls Shutdown to collect the result.
    constexpr s32 ORBIS_NP_BANDWIDTH_TEST_STATUS_FINISHED = 2;
    *status = ORBIS_NP_BANDWIDTH_TEST_STATUS_FINISHED;
    return ORBIS_OK;
}

static s32 PS4_SYSV_ABI sceNpBandwidthTestShutdown(s32 test_id, double* result) {
    if (test_id < 0 || result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // The result is two doubles (download/upload bandwidth). Offline completion has no samples.
    result[0] = 0.0;
    result[1] = 0.0;
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    ASSERT_MSG(Libraries::Kernel::sceKernelGetCompiledSdkVersion(&g_firmware_version) == ORBIS_OK,
               "Failed to get compiled SDK version.");
    g_shadnet_enabled = EmulatorSettings.IsShadNetEnabled();

    // Route live NP state changes from NpHandler into the dispatch queue so they are
    // delivered on the game's thread during sceNpCheckCallback (single delivery path).
    // Registered before Initialize() so the initial SignedIn events are captured.
    Libraries::Np::NpHandler::GetInstance().RegisterStateCallback(
        [](Libraries::UserService::OrbisUserServiceUserId user_id, OrbisNpState state) {
            QueueNpStateEvent(user_id, state);
        },
        nullptr);
    Libraries::Np::NpHandler::GetInstance().Initialize();
    EnsureLbp3HelperSignedInSticky("NP manager initialization");

    LIB_FUNCTION("GpLQDNKICac", "libSceNpManager", 1, "libSceNpManager", sceNpCreateRequest);
    LIB_FUNCTION("eiqMCt9UshI", "libSceNpManager", 1, "libSceNpManager", sceNpCreateAsyncRequest);
    LIB_FUNCTION("2rsFmlGWleQ", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpAvailability);
    LIB_FUNCTION("8Z2Jc5GvGDI", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpAvailabilityA);
    LIB_FUNCTION("KfGZg2y73oM", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpReachability);
    LIB_FUNCTION("r6MyYJkryz8", "libSceNpManager", 1, "libSceNpManager", sceNpCheckPlus);
    LIB_FUNCTION("KZ1Mj9yEGYc", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountLanguage);
    LIB_FUNCTION("TPMbgIxvog0", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountLanguageA);
    LIB_FUNCTION("ilwLM4zOmu4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetParentalControlInfo);
    LIB_FUNCTION("m9L3O6yst-U", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetParentalControlInfoA);
    LIB_FUNCTION("OzKvTvg3ZYU", "libSceNpManager", 1, "libSceNpManager", sceNpAbortRequest);
    LIB_FUNCTION("-QglDeRr8D8", "libSceNpManager", 1, "libSceNpManager", sceNpSetTimeout);
    LIB_FUNCTION("jyi5p9XWUSs", "libSceNpManager", 1, "libSceNpManager", sceNpWaitAsync);
    LIB_FUNCTION("uqcPJLWL08M", "libSceNpManager", 1, "libSceNpManager", sceNpPollAsync);
    LIB_FUNCTION("S7QTn72PrDw", "libSceNpManager", 1, "libSceNpManager", sceNpDeleteRequest);

    LIB_FUNCTION("Ghz9iWDUtC4", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountCountry);
    LIB_FUNCTION("JT+t00a3TxA", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountCountryA);
    LIB_FUNCTION("8VBTeRf1ZwI", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirth);
    LIB_FUNCTION("q3M7XzBKC3s", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirthA);
    LIB_FUNCTION("IPb1hd1wAGc", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatus);
    LIB_FUNCTION("oPO9U42YpgI", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatusA);
    LIB_FUNCTION("KO+11cgC7N0", "libSceNpManager", 1, "libSceNpManager",
                 sceNpSetGamePresenceOnline);
    LIB_FUNCTION("C0gNCiRIi4U", "libSceNpManager", 1, "libSceNpManager",
                 sceNpSetGamePresenceOnlineA);
    LIB_FUNCTION("uFJpaKNBAj4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterGamePresenceCallback);
    LIB_FUNCTION("KswxLxk4c1Y", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterGamePresenceCallbackA);
    LIB_FUNCTION("aJZyCcHxzu4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpUnregisterGamePresenceCallbackA);
    LIB_FUNCTION("Ybu6AxV6S0o", "libSceNpManager", 1, "libSceNpManager", sceNpIsPlusMember);
    LIB_FUNCTION("GImICnh+boA", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterPlusEventCallback);
    LIB_FUNCTION("xViqJdDgKl0", "libSceNpManager", 1, "libSceNpManager",
                 sceNpUnregisterPlusEventCallback);
    LIB_FUNCTION("e-ZuhGEoeC4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetNpReachabilityState);

    LIB_FUNCTION("a8R9-75u4iM", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountId);
    LIB_FUNCTION("rbknaUjpqWo", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountIdA);
    LIB_FUNCTION("p-o74CnoNzY", "libSceNpManager", 1, "libSceNpManager", sceNpGetNpId);
    LIB_FUNCTION("XDncXQIJUSk", "libSceNpManager", 1, "libSceNpManager", sceNpGetOnlineId);
    LIB_FUNCTION("eQH7nWPcAgc", "libSceNpManager", 1, "libSceNpManager", sceNpGetState);
    LIB_FUNCTION("VgYczPGB5ss", "libSceNpManager", 1, "libSceNpManager", sceNpGetUserIdByAccountId);
    LIB_FUNCTION("F6E4ycq9Dbg", "libSceNpManager", 1, "libSceNpManager", sceNpGetUserIdByOnlineId);
    LIB_FUNCTION("A2CQ3kgSopQ", "libSceNpManager", 1, "libSceNpManager",
                 sceNpSetContentRestriction);
    LIB_FUNCTION("Ec63y59l9tw", "libSceNpManager", 1, "libSceNpManager", sceNpSetNpTitleId);
    LIB_FUNCTION("Oad3rvY-NJQ", "libSceNpManager", 1, "libSceNpManager", sceNpHasSignedUp);
    LIB_FUNCTION("3Zl8BePTh9Y", "libSceNpManager", 1, "libSceNpManager", sceNpCheckCallback);
    LIB_FUNCTION("JELHf4xPufo", "libSceNpManager", 1, "libSceNpManager", sceNpCheckCallbackForLib);
    LIB_FUNCTION("VfRSmPmj8Q8", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterStateCallback);
    LIB_FUNCTION("mjjTXh+NHWY", "libSceNpManager", 1, "libSceNpManager",
                 sceNpUnregisterStateCallback);
    LIB_FUNCTION("qQJfO8HAiaY", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterStateCallbackA);
    LIB_FUNCTION("M3wFXbYQtAA", "libSceNpManager", 1, "libSceNpManager",
                 sceNpUnregisterStateCallbackA);
    LIB_FUNCTION("hw5KNqAAels", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterNpReachabilityStateCallback);
    LIB_FUNCTION("cRILAEvn+9M", "libSceNpManager", 1, "libSceNpManager",
                 sceNpUnregisterNpReachabilityStateCallback);
    LIB_FUNCTION("JELHf4xPufo", "libSceNpManagerForToolkit", 1, "libSceNpManager",
                 sceNpCheckCallbackForLib);
    LIB_FUNCTION("0c7HbXRKUt4", "libSceNpManagerForToolkit", 1, "libSceNpManager",
                 sceNpRegisterStateCallbackForToolkit);
    LIB_FUNCTION("YIvqqvJyjEc", "libSceNpManagerForToolkit", 1, "libSceNpManager",
                 sceNpUnregisterStateCallbackForToolkit);

    LIB_FUNCTION("BYIZGKm6bO4", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestGetStatus);
    LIB_FUNCTION("jktww3yJXnc", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestInitStart);
    LIB_FUNCTION("pLr1fEQS1z8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestShutdown);
    LIB_FUNCTION("8533Q+LU7EQ", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupCreateTitleCtx);
    LIB_FUNCTION("mtqDK9zkoIE", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupDeleteTitleCtx);

    LIB_FUNCTION("2rsFmlGWleQ", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpCheckNpAvailability);
    LIB_FUNCTION("a8R9-75u4iM", "libSceNpManagerCompat", 1, "libSceNpManager", sceNpGetAccountId);
    LIB_FUNCTION("KZ1Mj9yEGYc", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountLanguage);
    LIB_FUNCTION("Ghz9iWDUtC4", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountCountry);
    LIB_FUNCTION("8VBTeRf1ZwI", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirth);
    LIB_FUNCTION("IPb1hd1wAGc", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatus);
    LIB_FUNCTION("ilwLM4zOmu4", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetParentalControlInfo);
};

} // namespace Libraries::Np::NpManager
