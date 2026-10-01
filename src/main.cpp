// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>
#include <CLI/CLI.hpp>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_messagebox.h>

#include "common/arch.h"
#include "common/key_manager.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/debugger.h"
#include "core/emulator_settings.h"
#include "core/emulator_state.h"
#include "core/file_sys/fs.h"
#include "core/ipc/ipc.h"
#include "core/lbp3_online.h"
#include "core/performance_telemetry.h"
#include "core/user_settings.h"
#include "emulator.h"
#include "imgui/big_picture/big_picture.h"
#include "input/input_movie.h"
#include "video_core/renderdoc.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace {

std::optional<bool> ParseBoolean(std::string_view value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    return std::nullopt;
}

std::optional<std::pair<uint32_t, uint32_t>> ParseResolution(std::string_view value) {
    const auto separator = value.find_first_of("xX");
    if (separator == std::string_view::npos || separator == 0 || separator + 1 == value.size()) {
        return std::nullopt;
    }

    uint32_t width = 0;
    uint32_t height = 0;
    const auto width_result = std::from_chars(value.data(), value.data() + separator, width, 10);
    const auto height_result =
        std::from_chars(value.data() + separator + 1, value.data() + value.size(), height, 10);
    if (width_result.ec != std::errc{} || width_result.ptr != value.data() + separator ||
        height_result.ec != std::errc{} || height_result.ptr != value.data() + value.size() ||
        width < 640 || height < 360 || width > 7680 || height > 4320) {
        return std::nullopt;
    }
    return std::pair{width, height};
}

} // namespace

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

#if defined(__APPLE__) && defined(ARCH_X86_64)
    // KosmicKrisp only supports Apple Silicon. Check that we are not running on an Intel Mac.
    int sysctl_ret = 0;
    size_t sysctl_size = sizeof(sysctl_ret);
    sysctlbyname("sysctl.proc_translated", &sysctl_ret, &sysctl_size, nullptr, 0);
    if (sysctl_ret != 1) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "shadPS4",
                                 "shadPS4 only supports Apple Silicon Macs.", nullptr);
        std::cout << "shadPS4 only supports Apple Silicon Macs." << std::endl;
        return -1;
    }
#endif

    CLI::App app{"shadPS4 Emulator CLI"};

    // ---- CLI state ----
    std::optional<std::string> gamePath;
    std::vector<std::string> gameArgs;
    std::optional<std::filesystem::path> overrideRoot;
    std::optional<int> waitPid;
    bool waitForDebugger = false;

    std::optional<std::string> fullscreenStr;
    std::optional<std::string> resolutionStr;
    std::optional<std::string> lbp3PatchBubblesStr;
    std::optional<std::string> lbp3DisableSpriteLightsStr;
    std::optional<std::string> lbp3DisableToneMapStr;
    std::optional<std::string> lbp3DirectLevelStr;
    bool ignoreGamePatch = false;
    bool showFps = false;
    bool hideFps = false;
    bool configClean = false;
    bool configGlobal = false;
    bool bigPicture = false;
    bool perfTelemetry = false;
    std::optional<std::string> perfOutput;
    bool lbp3Online = false;

    std::optional<std::filesystem::path> addGameFolder;
    std::optional<std::filesystem::path> setAddonFolder;
    std::optional<std::string> patchFile;

    // ---- Options ----
    app.add_option("-g,--game", gamePath, "Game path or ID");
    app.add_option("-p,--patch", patchFile, "Patch file to apply");
    app.add_flag("-i,--ignore-game-patch", ignoreGamePatch,
                 "Disable automatic loading of game patches");

    app.add_flag("-b,--big-picture", bigPicture, "Start in Big Picture Mode");

    // FULLSCREEN: behavior-identical
    app.add_option("-f,--fullscreen", fullscreenStr, "Fullscreen mode (true|false)");
    app.add_option("--resolution", resolutionStr,
                   "Internal render and output window resolution (for example 1920x1080)");

    app.add_option("--override-root", overrideRoot)->check(CLI::ExistingDirectory);

    app.add_flag("--wait-for-debugger", waitForDebugger);
    app.add_option("--wait-for-pid", waitPid);

    app.add_flag("--show-fps", showFps);
    app.add_flag("--hide-fps", hideFps);
    app.add_flag("--config-clean", configClean);
    app.add_flag("--config-global", configGlobal);
    app.add_flag("--log-append", Common::Log::g_should_append);
    app.add_flag("--perf-telemetry", perfTelemetry,
                 "Record detailed per-frame performance telemetry CSV files");
    app.add_option("--perf-output", perfOutput,
                   "Existing directory for this session's performance telemetry")
        ->check(CLI::ExistingDirectory);
    app.add_flag("--lbp3-online", lbp3Online,
                 "Enable the local LBP3 helper backend and P2P transport");
    app.add_option("--lbp3-patch-bubbles", lbp3PatchBubblesStr,
                   "Enable the LBP3 prize-bubble compatibility patch (true|false)");
    app.add_option("--lbp3-disable-sprite-lights", lbp3DisableSpriteLightsStr,
                   "Disable LBP3 sprite lights (true|false)");
    app.add_option("--lbp3-disable-tone-map", lbp3DisableToneMapStr,
                   "Disable the LBP3 sprite-light tone-map pass (true|false)");
    app.add_option("--lbp3-direct-level", lbp3DirectLevelStr,
                   "Directly load an LBP3 Story/DLC LevelID "
                   "(slot_type:slot_id[:adventure_type:adventure_id])");

    app.add_option("--add-game-folder", addGameFolder)->check(CLI::ExistingDirectory);
    app.add_option("--set-addon-folder", setAddonFolder)->check(CLI::ExistingDirectory);

    // ---- Capture args after `--` verbatim ----
    app.allow_extras();
    app.parse_complete_callback([&]() {
        const auto& extras = app.remaining();
        if (!extras.empty()) {
            gameArgs = extras;
        }
    });

    // ---- No-args behavior ----
    if (argc == 1) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "shadPS4",
                                 "This is a CLI application. Please use the QTLauncher for a GUI:\n"
                                 "https://github.com/shadps4-emu/shadps4-qtlauncher/releases",
                                 nullptr);
        std::cout << app.help();
        return -1;
    }

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    Core::PerfTelemetry::SetStartRequested(perfTelemetry);
    if (perfOutput) {
        Core::PerfTelemetry::SetOutputDirectory(*perfOutput);
    }

    if (waitPid)
        Core::Debugger::WaitForPid(*waitPid);

    // Initialize main log with default config
    Common::Log::Setup("shadps4.log");

    if (!lbp3DirectLevelStr) {
        const char* direct_level = std::getenv("SHADPS4_LBP3_DIRECT_LEVEL");
        if (direct_level == nullptr || direct_level[0] == '\0') {
            direct_level = std::getenv("SHADPS4_DIAGNOSTIC_LBP3_DIRECT_LEVEL");
        }
        if (direct_level != nullptr && direct_level[0] != '\0') {
            lbp3DirectLevelStr = direct_level;
        }
    }
    if (lbp3DirectLevelStr && !MemoryPatcher::ConfigureLbp3DirectLevel(*lbp3DirectLevelStr)) {
        LOG_ERROR(Debug,
                  "Invalid --lbp3-direct-level value '{}'; expected "
                  "slot_type:slot_id[:adventure_type:adventure_id]",
                  *lbp3DirectLevelStr);
        return 1;
    }

    Core::Lbp3Online::SetEnabled(lbp3Online);
    if (lbp3Online) {
        LOG_INFO(Debug, "LBP3 online helper mode requested");
    }

    LOG_INFO(Debug, "Run: {}", std::span(argv, argc));

    IPC::Instance().Init();

    auto emu_state = std::make_shared<EmulatorState>();
    EmulatorState::SetInstance(emu_state);
    UserSettings.Load();

    // Initialize key manager
    auto key_manager = KeyManager::GetInstance();
    key_manager->LoadFromFile();

    // Load configurations
    std::shared_ptr<EmulatorSettingsImpl> emu_settings = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(emu_settings);
    emu_settings->Load();

    const auto input_path = [](const char* name) -> std::filesystem::path {
        const char* value = std::getenv(name);
        return value ? std::filesystem::path{value} : std::filesystem::path{};
    };
    const auto input_record = input_path("SHADPS4_INPUT_RECORD");
    const auto input_replay = input_path("SHADPS4_INPUT_REPLAY");
    const auto input_report = input_path("SHADPS4_INPUT_REPORT");
    const auto replay_home = input_path("SHADPS4_REPLAY_HOME");
    const auto input_screenshots = input_path("SHADPS4_INPUT_SCREENSHOTS");
    if ((!input_record.empty() || !input_replay.empty()) && !input_screenshots.empty()) {
        std::filesystem::create_directories(input_screenshots);
        Common::FS::SetUserPath(Common::FS::PathType::ScreenshotsDir, input_screenshots);
    }
    if (!input_replay.empty()) {
        // Replays write only to a fresh working copy, never to the live profile.
        if (replay_home.empty() || !std::filesystem::is_directory(replay_home) ||
            std::filesystem::equivalent(replay_home, EmulatorSettings.GetHomeDir())) {
            std::cerr << "Controller replay requires a separate existing SHADPS4_REPLAY_HOME\n";
            return 1;
        }
        EmulatorSettings.SetRuntimeHomeDir(replay_home);
    } else if (!replay_home.empty()) {
        std::cerr << "SHADPS4_REPLAY_HOME requires an input replay\n";
        return 1;
    }
    std::string input_error;
    if (!Input::Movie::Configure(
            input_record, input_replay, input_report,
            [] {
                SDL_Event event{};
                event.type = SDL_EVENT_QUIT;
                SDL_PushEvent(&event);
            },
            [] { VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly); },
            input_error)) {
        std::cerr << input_error << '\n';
        return 1;
    }

    // Configure logger appropriately
    Common::Log::g_should_append |= EmulatorSettings.IsLogAppend();

    if (bigPicture) {
        BigPictureMode::Launch(argv[0]);
        return 0;
    }

    // ---- Utility commands ----
    if (addGameFolder) {
        EmulatorSettings.AddGameInstallDir(*addGameFolder);
        EmulatorSettings.Save();
        LOG_INFO(Config, "Game folder successfully saved.");
        return 0;
    }

    if (setAddonFolder) {
        EmulatorSettings.SetAddonInstallDir(*setAddonFolder);
        EmulatorSettings.Save();
        LOG_INFO(Config, "Addon folder successfully saved.");
        if (!gamePath.has_value() && gameArgs.empty()) {
            return 0;
        }
    }

    if (!gamePath.has_value()) {
        if (!gameArgs.empty()) {
            gamePath = gameArgs.front();
            gameArgs.erase(gameArgs.begin());
        } else {
            LOG_ERROR(Debug, "Please provide a game path or ID.");
            return 1;
        }
    }
    if (!gameArgs.empty()) {
        if (gameArgs.front() == "--") {
            gameArgs.erase(gameArgs.begin());
        } else {
            LOG_ERROR(Debug, "unhandled flags");
            return 1;
        }
    }

    // ---- Apply flags ----
    if (patchFile)
        MemoryPatcher::patch_file = *patchFile;

    if (ignoreGamePatch)
        Core::FileSys::MntPoints::ignore_game_patches = true;

    std::optional<bool> fullscreenOverride;
    if (fullscreenStr) {
        fullscreenOverride = ParseBoolean(*fullscreenStr);
        if (!fullscreenOverride.has_value()) {
            LOG_ERROR(Debug, "Invalid argument for --fullscreen (use true|false)");
            return 1;
        }
        EmulatorSettings.SetFullScreen(*fullscreenOverride);
    }

    std::optional<std::pair<uint32_t, uint32_t>> resolutionOverride;
    if (resolutionStr) {
        resolutionOverride = ParseResolution(*resolutionStr);
        if (!resolutionOverride.has_value()) {
            LOG_ERROR(Debug,
                      "Invalid argument for --resolution (use WIDTHxHEIGHT, 640x360..7680x4320)");
            return 1;
        }
        const auto [width, height] = *resolutionOverride;
        EmulatorSettings.SetInternalScreenWidth(width);
        EmulatorSettings.SetInternalScreenHeight(height);
        EmulatorSettings.SetWindowWidth(width);
        EmulatorSettings.SetWindowHeight(height);
    }

    const auto apply_boolean_option = [](const std::optional<std::string>& option,
                                         std::string_view option_name, bool& destination) {
        if (!option.has_value()) {
            return true;
        }
        const auto value = ParseBoolean(*option);
        if (!value.has_value()) {
            LOG_ERROR(Debug, "Invalid argument for {} (use true|false)", option_name);
            return false;
        }
        destination = *value;
        return true;
    };
    if (!apply_boolean_option(lbp3PatchBubblesStr, "--lbp3-patch-bubbles",
                              MemoryPatcher::g_lbp3_patch_prize_bubbles) ||
        !apply_boolean_option(lbp3DisableSpriteLightsStr, "--lbp3-disable-sprite-lights",
                              MemoryPatcher::g_lbp3_disable_sprite_lights) ||
        !apply_boolean_option(lbp3DisableToneMapStr, "--lbp3-disable-tone-map",
                              MemoryPatcher::g_lbp3_disable_tone_map)) {
        return 1;
    }

    std::optional<bool> showFpsOverride;
    if (showFps && hideFps) {
        LOG_ERROR(Debug, "--show-fps and --hide-fps cannot be used together");
        return 1;
    }
    if (showFps || hideFps) {
        showFpsOverride = showFps;
        EmulatorSettings.SetShowFpsCounter(*showFpsOverride);
    }

    if (configClean)
        EmulatorSettings.SetConfigMode(ConfigMode::Clean);

    if (configGlobal)
        EmulatorSettings.SetConfigMode(ConfigMode::Global);

    // ---- Resolve game path or ID ----
    std::filesystem::path ebootPath(*gamePath);
    if (!std::filesystem::exists(ebootPath)) {
        bool found = false;
        constexpr int maxDepth = 5;
        for (const auto& installDir : EmulatorSettings.GetGameInstallDirs()) {
            if (auto foundPath = Common::FS::FindGameByID(installDir, *gamePath, maxDepth)) {
                ebootPath = *foundPath;
                found = true;
                break;
            }
        }
        if (!found) {
            LOG_ERROR(Debug, "Game ID or file path not found: {}", *gamePath);
            return 1;
        }
    }

    auto* emulator = Common::Singleton<Core::Emulator>::Instance();
    emulator->executableName = argv[0];
    emulator->waitForDebuggerBeforeRun = waitForDebugger;
    emulator->fullscreenOverride = fullscreenOverride;
    emulator->showFpsOverride = showFpsOverride;
    emulator->resolutionOverride = resolutionOverride;
    emulator->Run(ebootPath, gameArgs, overrideRoot);

    return 0;
}
