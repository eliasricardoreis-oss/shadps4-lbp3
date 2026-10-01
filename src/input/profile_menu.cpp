// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "input/profile_menu.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <string>
#include <SDL3/SDL.h>
#include <imgui.h>
#include "common/singleton.h"
#include "core/user_settings.h"
#include "imgui/imgui_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "input/controller.h"
#include "input/input_handler.h"

namespace Input::Profiles {
namespace {
struct Profile {
    std::string name;
    std::string controller;
    SDL_JoystickID device{};
    bool available{};
};
std::mutex mutex;
std::array<Profile, 4> profiles;
SDL_JoystickID selected_device{};
u8 selected_profile{};
bool open{};
int stick_direction{};

enum Action { SelectDevice = 1, SelectProfile, Assign, Close };
u32 CommandEvent() {
    static const u32 type = SDL_RegisterEvents(1);
    return type;
}
void Send(Action action, SDL_JoystickID device = 0, u8 slot = 0) {
    SDL_Event event{};
    event.type = CommandEvent();
    event.user.code = action;
    event.user.data1 = reinterpret_cast<void*>(uintptr_t(device));
    event.user.data2 = reinterpret_cast<void*>(uintptr_t(slot));
    SDL_PushEvent(&event);
}
void CloseMenu() {
    ReleaseAllInputs();
    std::lock_guard lock(mutex);
    if (!open)
        return;
    open = false;
    stick_direction = 0;
    ImGui::Core::ReleaseGamepadInputCapture();
}
// Called with mutex held. Selection follows physical input, never the profile
// slot whose label happened to be selected by a different controller.
bool SelectDeviceLocked(SDL_JoystickID device) {
    if (!device || selected_device == device)
        return false;
    for (u8 i = 0; i < profiles.size(); ++i) {
        if (profiles[i].device == device) {
            selected_device = device;
            selected_profile = i;
            stick_direction = 0;
            return true;
        }
    }
    return false;
}
bool KnownDevice(SDL_JoystickID device) {
    return Common::Singleton<GameControllers>::Instance()->GetGamepadIndexFromJoystickId(device) <
           4;
}
void MoveProfile(int direction) {
    for (int n = 0; n < 4; ++n) {
        selected_profile = (selected_profile + direction + 4) % 4;
        if (profiles[selected_profile].available)
            break;
    }
}
std::string DeviceName(SDL_Gamepad* pad) {
    switch (SDL_GetRealGamepadType(pad)) {
    case SDL_GAMEPAD_TYPE_PS5:
        return "DualSense";
    case SDL_GAMEPAD_TYPE_PS4:
        return "DUALSHOCK 4";
    default: {
        const char* name = SDL_GetGamepadName(pad);
        return name && *name ? name : "Геймпад";
    }
    }
}
class Menu final : public ImGui::Layer {
public:
    void Draw() override {
        std::lock_guard lock(mutex);
        if (!open)
            return;
        const auto& io = ImGui::GetIO();
        const float width = std::min(700.0f, std::max(300.0f, io.DisplaySize.x - 32.0f));
        ImGui::SetNextWindowPos({io.DisplaySize.x / 2, io.DisplaySize.y / 2}, ImGuiCond_Always,
                                {0.5f, 0.5f});
        ImGui::SetNextWindowSizeConstraints({width, 0}, {width, io.DisplaySize.y - 32});
        ImGui::SetNextWindowBgAlpha(1);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 1));
        // SDL handles navigation with device identity; ImGui only handles the mouse.
        if (ImGui::Begin("Профиль игрока", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                             ImGuiWindowFlags_NoMove)) {
            const auto count =
                std::ranges::count_if(profiles, [](const Profile& p) { return p.device != 0; });
            ImGui::Text("Подключено геймпадов: %d", int(count));
            for (const auto& profile : profiles) {
                if (!profile.device)
                    continue;
                ImGui::PushID(int(profile.device));
                const std::string label = profile.controller + "  |  " + profile.name;
                if (ImGui::Selectable(label.c_str(), selected_device == profile.device))
                    Send(SelectDevice, profile.device);
                ImGui::PopID();
            }
            if (!selected_device) {
                ImGui::TextWrapped("Нет подключённых геймпадов.");
            } else {
                ImGui::TextWrapped("Нажмите кнопку на нужном геймпаде, чтобы выбрать его.");
                ImGui::Separator();
                const auto it = std::ranges::find(profiles, selected_device, &Profile::device);
                ImGui::TextWrapped("Профиль для %s:",
                                   it == profiles.end() ? "геймпада" : it->controller.c_str());
                for (u8 i = 0; i < profiles.size(); ++i) {
                    const auto& profile = profiles[i];
                    if (!profile.available)
                        continue;
                    ImGui::PushID(100 + i);
                    std::string label = profile.name;
                    if (profile.device == selected_device)
                        label += "  (текущий)";
                    else if (profile.device)
                        label += "  |  " + profile.controller;
                    else
                        label += "  (свободен)";
                    if (ImGui::Selectable(label.c_str(), selected_profile == i))
                        Send(SelectProfile, 0, i);
                    ImGui::PopID();
                }
                const auto& target = profiles[selected_profile];
                if (target.device && target.device != selected_device && it != profiles.end()) {
                    ImGui::TextWrapped("%s перейдёт к профилю %s.", target.controller.c_str(),
                                       it->name.c_str());
                }
                ImGui::Separator();
                ImGui::BeginDisabled(!target.available);
                if (ImGui::Button("Выбрать и играть"))
                    Send(Assign, selected_device, selected_profile);
                ImGui::EndDisabled();
                ImGui::SameLine();
            }
            if (ImGui::Button("Закрыть"))
                Send(Close);
            ImGui::TextWrapped("Вверх / вниз: профиль. X / A: выбрать. Круг / B: закрыть.");
        }
        ImGui::End();
        ImGui::PopStyleColor();
    }
} menu;
} // namespace

bool IsOpen() {
    std::lock_guard lock(mutex);
    return open;
}
void Refresh() {
    auto& controllers = *Common::Singleton<GameControllers>::Instance();
    std::lock_guard lock(mutex);
    bool found = false;
    for (u8 i = 0; i < profiles.size(); ++i) {
        auto* user = UserManagement.GetUserByPlayerIndex(i + 1);
        auto* pad = controllers[i]->m_sdl_gamepad;
        // A non-null SDL handle can outlive a disconnected device.
        const bool connected = pad && SDL_GamepadConnected(pad);
        profiles[i] = {user ? user->user_name : "", connected ? DeviceName(pad) : "",
                       connected ? SDL_GetGamepadID(pad) : 0, user != nullptr};
        found |= selected_device != 0 && profiles[i].device == selected_device;
    }
    // Two physical devices of the same model are valid. Give them distinct
    // labels instead of deduplicating by name, model or VID/PID.
    const auto names = profiles;
    for (auto& profile : profiles) {
        if (!profile.device)
            continue;
        int count = 0, index = 1;
        for (const auto& other : names) {
            if (other.device && other.controller == profile.controller) {
                ++count;
                if (other.device < profile.device)
                    ++index;
            }
        }
        if (count > 1)
            profile.controller += " #" + std::to_string(index);
    }
    if (!found) {
        selected_device = 0;
        stick_direction = 0;
        for (const auto& profile : profiles) {
            if (SelectDeviceLocked(profile.device))
                break;
        }
    }
}
void Open(SDL_JoystickID device) {
    Refresh();
    ReleaseAllInputs();
    std::lock_guard lock(mutex);
    SelectDeviceLocked(device);
    if (!open) {
        const auto current = std::ranges::find(profiles, selected_device, &Profile::device);
        if (selected_device && current != profiles.end())
            selected_profile = u8(current - profiles.begin());
        stick_direction = 0;
        ImGui::Core::AcquireGamepadInputCapture();
        open = true;
    }
}
bool ProcessEvent(const SDL_Event& event) {
    if (event.type == CommandEvent()) {
        if (!IsOpen())
            return true;
        const auto device = SDL_JoystickID(reinterpret_cast<uintptr_t>(event.user.data1));
        const auto slot = uintptr_t(event.user.data2);
        if (event.user.code == Close)
            CloseMenu();
        else if (event.user.code == SelectDevice)
            Open(device);
        else if (event.user.code == SelectProfile && slot < 4) {
            std::lock_guard lock(mutex);
            if (profiles[slot].available)
                selected_profile = slot;
        } else if (event.user.code == Assign && slot < 4) {
            ReleaseAllInputs();
            if (Common::Singleton<GameControllers>::Instance()->AssignDeviceToProfile(device,
                                                                                      slot)) {
                Refresh();
                CloseMenu();
            }
        }
        return true;
    }
    const bool guide = (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN ||
                        event.type == SDL_EVENT_GAMEPAD_BUTTON_UP) &&
                       event.gbutton.button == SDL_GAMEPAD_BUTTON_GUIDE;
    const bool shortcut = (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) &&
                          event.key.key == SDLK_F2;
    if (guide || shortcut) {
        if (guide && !KnownDevice(event.gbutton.which))
            return true;
        if ((guide && event.gbutton.down) || (shortcut && event.key.down && !event.key.repeat)) {
            bool same_device;
            {
                std::lock_guard lock(mutex);
                same_device = !guide || selected_device == event.gbutton.which;
            }
            if (IsOpen() && same_device)
                CloseMenu();
            else
                Open(guide ? event.gbutton.which : 0);
        }
        return true;
    }
    if (!IsOpen())
        return false;
    if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
        const bool keyboard = event.type == SDL_EVENT_KEY_DOWN;
        if (!keyboard && !KnownDevice(event.gbutton.which))
            return true;
        if (keyboard && event.key.repeat)
            return true;
        const auto key = keyboard ? event.key.key : SDLK_UNKNOWN;
        const auto button =
            keyboard ? SDL_GAMEPAD_BUTTON_INVALID : SDL_GamepadButton(event.gbutton.button);
        std::lock_guard lock(mutex);
        const bool switched = !keyboard && SelectDeviceLocked(event.gbutton.which);
        // The first press on another pad selects it; it must not also confirm
        // the profile previously highlighted for a different device.
        if (switched)
            return true;
        if (key == SDLK_ESCAPE || button == SDL_GAMEPAD_BUTTON_EAST)
            Send(Close);
        else if (key == SDLK_RETURN || button == SDL_GAMEPAD_BUTTON_SOUTH)
            Send(Assign, selected_device, selected_profile);
        else if (key == SDLK_UP || button == SDL_GAMEPAD_BUTTON_DPAD_UP)
            MoveProfile(-1);
        else if (key == SDLK_DOWN || button == SDL_GAMEPAD_BUTTON_DPAD_DOWN)
            MoveProfile(1);
        return true;
    }
    if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION) {
        if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY && KnownDevice(event.gaxis.which)) {
            const int direction = event.gaxis.value < -18000  ? -1
                                  : event.gaxis.value > 18000 ? 1
                                                              : 0;
            std::lock_guard lock(mutex);
            // Neutral/noisy input from another controller cannot steal focus.
            if (selected_device != event.gaxis.which) {
                if (direction)
                    SelectDeviceLocked(event.gaxis.which);
            } else {
                if (direction && direction != stick_direction)
                    MoveProfile(direction);
                stick_direction = direction;
            }
        }
        return true;
    }
    return event.type == SDL_EVENT_KEY_UP || event.type == SDL_EVENT_TEXT_INPUT ||
           event.type == SDL_EVENT_GAMEPAD_BUTTON_UP ||
           event.type == SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN ||
           event.type == SDL_EVENT_GAMEPAD_TOUCHPAD_UP ||
           event.type == SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION ||
           event.type == SDL_EVENT_GAMEPAD_SENSOR_UPDATE;
}
void Register() {
    ImGui::Layer::AddLayer(&menu);
}
void Unregister() {
    std::lock_guard lock(mutex);
    if (open)
        ImGui::Core::ReleaseGamepadInputCapture();
    open = false;
    ImGui::Layer::RemoveLayer(&menu);
}
} // namespace Input::Profiles
