// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/memory_patcher.h"
#include "common/signal_context.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/signals.h"
#include "emulator.h"

#include <atomic>

#ifdef _WIN32
#include <windows.h>
static constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
extern std::array<OrbisKernelExceptionHandler, 32> Handlers;
} // namespace Libraries::Kernel
#endif

namespace Core {

#if defined(__APPLE__) && defined(ARCH_X86_64)
namespace {

std::atomic_bool lbp3_direct_level_consumed{};
std::atomic_bool lbp3_direct_level_loader_configured{};
std::atomic_bool lbp3_direct_level_configured{};
std::atomic_bool lbp3_direct_level_wait_reported{};
std::atomic_uint lbp3_direct_level_attempts{};

} // namespace
#endif

#if defined(_WIN32)

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    bool handled = false;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    case MS_VC_EXCEPTION:
        LOG_DEBUG(Debug, "Pass MS_VC_EXCEPTION at {} to handler", address);
        return EXCEPTION_EXECUTE_HANDLER;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // Breakpoints almost certainly come from our asserts/unreachables, no need to log it again.
    if (code != EXCEPTION_BREAKPOINT) {
        LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {}", code, address);
        Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

#else

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
#if defined(__APPLE__) && defined(ARCH_X86_64)
        // CUSA00063 01.26's Game::Update normally calls GetLevelManager here. The built-in
        // direct-level patch replaces that call with UD2. Once the profile and manager are ready,
        // invoke the same high-level launcher used by the game UI with a known registered carrier
        // slot. A later hook replaces only the final LaunchConfig LevelID, after progression
        // lookup.
        if (code_address == reinterpret_cast<void*>(0x40bf01) &&
            MemoryPatcher::g_lbp3_direct_level) {
            const auto level = *MemoryPatcher::g_lbp3_direct_level;
            auto& state = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext->__ss;
            const auto resume_manager_getter = [&state] {
                state.__rsp -= sizeof(u64);
                *reinterpret_cast<u64*>(state.__rsp) = 0x40bf06;
                state.__rip = 0x93b060;
            };

            constexpr uintptr_t ManagerGlobal = 0x17abbd8;
            constexpr uintptr_t GameGlobal = 0x16ecd80;
            const u64 manager = *reinterpret_cast<const u64*>(ManagerGlobal);
            const u64 game = *reinterpret_cast<const u64*>(GameGlobal);
            const bool profile_ready = game != 0 && *reinterpret_cast<const u8*>(game + 0x4f5) != 0;
            const bool manager_idle =
                manager != 0 && *reinterpret_cast<const u32*>(manager + 0x1380) == 0;
            if (!profile_ready || !manager_idle) {
                if (!lbp3_direct_level_wait_reported.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Debug,
                             "[LBP3_DIRECT_LEVEL] state=waiting profile_ready={} manager={:#x} "
                             "manager_idle={}",
                             profile_ready, manager, manager_idle);
                }
                resume_manager_getter();
                return;
            }

            if (!lbp3_direct_level_consumed.exchange(true, std::memory_order_relaxed)) {
                constexpr u32 CarrierSlotType = 2;
                constexpr u32 CarrierSlotId = 55;
                const u32 attempt =
                    lbp3_direct_level_attempts.fetch_add(1, std::memory_order_relaxed) + 1;

                // Keep the carrier LevelID alive on the guest stack until the launcher returns.
                state.__rsp -= sizeof(MemoryPatcher::Lbp3DirectLevelTarget);
                auto* carrier =
                    reinterpret_cast<MemoryPatcher::Lbp3DirectLevelTarget*>(state.__rsp);
                *carrier = {.slot_type = CarrierSlotType, .slot_id = CarrierSlotId};
                state.__rsp -= sizeof(u64);
                *reinterpret_cast<u64*>(state.__rsp) = 0x9444ba;
                state.__rdi = manager;
                state.__rsi = reinterpret_cast<u64>(carrier);
                state.__rdx = 1;
                state.__rcx = 0;
                state.__r8 = 0;
                state.__rip = 0x943310;
                LOG_INFO(Debug,
                         "[LBP3_DIRECT_LEVEL] state=launch attempt={} target={}:{} "
                         "adventure={}:{} carrier={}:{} manager={:#x}",
                         attempt, level.slot_type, level.slot_id, level.adventure_type,
                         level.adventure_id, CarrierSlotType, CarrierSlotId, manager);
                return;
            }

            resume_manager_getter();
            return;
        }

        // Force the high-level carrier launch through mode 1 and prevent it from replacing the
        // supplied carrier with the current Pod LevelID. Reproduce the displaced function prologue.
        if (code_address == reinterpret_cast<void*>(0x9444c0) &&
            MemoryPatcher::g_lbp3_direct_level) {
            const auto level = *MemoryPatcher::g_lbp3_direct_level;
            auto& state = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext->__ss;
            const auto return_address = *reinterpret_cast<const u64*>(state.__rsp);
            if (return_address == 0x9436db &&
                !lbp3_direct_level_loader_configured.exchange(true, std::memory_order_relaxed)) {
                const auto* carrier = reinterpret_cast<const u32*>(state.__rsi);
                const u32 carrier_type = carrier != nullptr ? carrier[0] : 0;
                const u32 carrier_id = carrier != nullptr ? carrier[1] : 0;
                state.__rcx = 1;
                state.__r9 = 0;
                LOG_INFO(Debug,
                         "[LBP3_DIRECT_LEVEL] state=loader target={}:{} adventure={}:{} "
                         "carrier={}:{} mode={} use_current_level={}",
                         level.slot_type, level.slot_id, level.adventure_type, level.adventure_id,
                         carrier_type, carrier_id, state.__rcx, state.__r9);
            }

            state.__rsp -= sizeof(u64);
            *reinterpret_cast<u64*>(state.__rsp) = state.__rbp;
            state.__rbp = state.__rsp;
            state.__rip = 0x9444c4;
            return;
        }

        // Return probe for the direct high-level launcher. Restore the GetLevelManager result and
        // original Game::Update continuation after dropping the carrier reserved above.
        if (code_address == reinterpret_cast<void*>(0x9444ba) &&
            MemoryPatcher::g_lbp3_direct_level) {
            auto& state = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext->__ss;
            constexpr uintptr_t ManagerGlobal = 0x17abbd8;
            const u64 manager = *reinterpret_cast<const u64*>(ManagerGlobal);
            const u32 manager_loading =
                manager != 0 ? *reinterpret_cast<const u32*>(manager + 0x1380) : 0;
            LOG_INFO(Debug, "[LBP3_DIRECT_LEVEL] state=returned result={} manager_loading={}",
                     static_cast<u32>(state.__rax & 0xff), manager_loading);
            state.__rsp += sizeof(MemoryPatcher::Lbp3DirectLevelTarget);
            state.__rsp -= sizeof(u64);
            *reinterpret_cast<u64*>(state.__rsp) = 0x40bf06;
            state.__rip = 0x93b060;
            return;
        }

        // LaunchConfig is created after the carrier's resource vector has been resolved. Replace
        // its LevelID only at the proven high-level callsite, so locked Story/DLC targets never go
        // through the progression-gated lookup. Reproduce the displaced function prologue.
        if (code_address == reinterpret_cast<void*>(0xc7c2b0) &&
            MemoryPatcher::g_lbp3_direct_level) {
            const auto level = *MemoryPatcher::g_lbp3_direct_level;
            auto& state = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext->__ss;
            auto* slot_id = reinterpret_cast<u32*>(state.__rdx);
            const auto* resource = reinterpret_cast<const u32*>(state.__rsi);
            const auto return_address = *reinterpret_cast<const u64*>(state.__rsp);
            if (slot_id != nullptr && resource != nullptr && return_address == 0x944d65 &&
                lbp3_direct_level_loader_configured.load(std::memory_order_relaxed) &&
                !lbp3_direct_level_configured.exchange(true, std::memory_order_relaxed)) {
                const u32 resolved_carrier_type = slot_id[0];
                const u32 resolved_carrier_id = slot_id[1];
                slot_id[0] = level.slot_type;
                slot_id[1] = level.slot_id;
                slot_id[2] = level.adventure_type;
                slot_id[3] = level.adventure_id;
                state.__rcx = 1;
                LOG_INFO(Debug,
                         "[LBP3_DIRECT_LEVEL] state=config target={}:{} adventure={}:{} mode={} "
                         "resolved_carrier={}:{}",
                         level.slot_type, level.slot_id, level.adventure_type, level.adventure_id,
                         state.__rcx, resolved_carrier_type, resolved_carrier_id);
            }

            state.__rsp -= sizeof(u64);
            *reinterpret_cast<u64*>(state.__rsp) = state.__rbp;
            state.__rbp = state.__rsp;
            state.__rip = 0xc7c2b4;
            return;
        }
#endif
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
