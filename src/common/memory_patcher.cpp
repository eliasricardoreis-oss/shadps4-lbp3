// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <charconv>
#include <codecvt>
#include <cstdlib>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include "common/arch.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_state.h"
#include "core/file_format/psf.h"
#include "core/performance_telemetry.h"
#include "memory_patcher.h"

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
std::string g_game_serial;
std::string patch_file;
bool g_lbp3_patch_prize_bubbles = true;
bool g_lbp3_disable_sprite_lights = false;
bool g_lbp3_disable_tone_map = false;
std::optional<Lbp3DirectLevelTarget> g_lbp3_direct_level;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;

static bool ParseU32(std::string_view text, uint32_t& value) {
    if (text.empty()) {
        return false;
    }
    int base = 10;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2);
        base = 16;
        if (text.empty()) {
            return false;
        }
    }
    uint32_t parsed{};
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, parsed, base);
    if (error != std::errc{} || position != end) {
        return false;
    }
    value = parsed;
    return true;
}

bool ConfigureLbp3DirectLevel(std::string_view spec) {
    std::array<std::string_view, 4> fields{};
    size_t field_count{};
    size_t begin{};
    while (begin <= spec.size() && field_count < fields.size()) {
        const size_t separator = spec.find(':', begin);
        fields[field_count++] = spec.substr(begin, separator - begin);
        if (separator == std::string_view::npos) {
            begin = spec.size() + 1;
            break;
        }
        begin = separator + 1;
    }
    if ((field_count != 2 && field_count != 4) || begin <= spec.size()) {
        return false;
    }

    Lbp3DirectLevelTarget target{};
    if (!ParseU32(fields[0], target.slot_type) || !ParseU32(fields[1], target.slot_id)) {
        return false;
    }
    if (field_count == 4 && (!ParseU32(fields[2], target.adventure_type) ||
                             !ParseU32(fields[3], target.adventure_id))) {
        return false;
    }
    g_lbp3_direct_level = target;
    return true;
}

std::string GetLbp3DirectLevelSpec() {
    if (!g_lbp3_direct_level) {
        return {};
    }
    const auto& target = *g_lbp3_direct_level;
    std::string spec = std::to_string(target.slot_type) + ":" + std::to_string(target.slot_id);
    if (target.adventure_type != 0 || target.adventure_id != 0) {
        spec +=
            ":" + std::to_string(target.adventure_type) + ":" + std::to_string(target.adventure_id);
    }
    return spec;
}

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

static bool isHexSym(const std::string& s) {
    return (s.size() >= 2 && (s[0] == '$' || s[0] == '#'));
}

static bool isHex0x(const std::string& s) {
    return (s.size() >= 3 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'));
}

// possible prefixes == (s == `0x` || s == `#` || s == `$`) will be interpet as hex value, else is
// decimal
static int convertNumBase(const std::string& s) {
    return (isHex0x(s) || isHexSym(s)) ? 16 : 10;
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(std::byteswap(floatUnion.i), sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(std::byteswap(doubleUnion.i), sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        // convert to little endian
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

static void ApplyBuiltInLbp3CompatibilityPatches() {
    if (g_game_serial != "CUSA00063") {
        return;
    }

    const auto* param_sfo = Common::Singleton<PSF>::Instance();
    const auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");
    if (app_version != "01.26") {
        LOG_WARNING(Loader,
                    "LBP3 built-in compatibility patches require app version 01.26; found {}",
                    app_version);
        return;
    }

    // These exact v1.26 signatures expose the small set of compatibility controls used by the
    // dedicated LBP3 launcher. Native sprite lights and tone mapping stay enabled by default.
    static constexpr std::string_view PickupSignature =
        "48 8d 1d 92 3a c5 00 80 3b 00 0f 84 c6 01 00 00";
    static constexpr std::string_view SpriteLightsSignature =
        "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 81 ec b8 03 00 00 48 89 fb 48 89 9d "
        "98 fc ff ff";
    static constexpr std::string_view ToneMapSpriteLightsSignature =
        "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 81 ec 68 01 00 00 48 8b 05 d5 2a c9 "
        "00";

    const auto apply_mask_patch = [](std::string_view signature, std::string_view name,
                                     std::string_view value, int mask_offset) {
        if (PatternScan(std::string{signature}) == 0) {
            LOG_ERROR(Loader,
                      "LBP3 compatibility patch '{}' skipped: v1.26 signature did not match", name);
            return;
        }
        PatchMemory(patchInfo{
            .gameSerial = "CUSA00063",
            .modNameStr = std::string{name},
            .offsetStr = std::string{signature},
            .valueStr = std::string{value},
            .isOffset = false,
            .littleEndian = false,
            .patchMask = PatchMask::Mask,
            .maskOffset = mask_offset,
        });
    };

    if (g_lbp3_patch_prize_bubbles) {
        apply_mask_patch(PickupSignature, "LBP3 built-in disable pickup resource blob",
                         "e9c701000090", 10);
    }
    if (g_lbp3_disable_sprite_lights) {
        apply_mask_patch(SpriteLightsSignature, "LBP3 built-in disable sprite lights", "c3", 0);
    }
    if (g_lbp3_disable_tone_map) {
        apply_mask_patch(ToneMapSpriteLightsSignature,
                         "LBP3 built-in disable sprite-light tone map", "c3", 0);
    }

    if (g_lbp3_direct_level) {
#if defined(__APPLE__) && defined(ARCH_X86_64)
        static constexpr uintptr_t GuestImageBase = 0x400000;
        static constexpr std::array<u8, 5> GameUpdateCallExpected{0xe8, 0x5a, 0xf1, 0x52, 0x00};
        static constexpr std::array<u8, 5> GameUpdateCallPatch{0x0f, 0x0b, 0x90, 0x90, 0x90};
        static constexpr std::array<u8, 2> ReturnProbeExpected{0x90, 0x0f};
        static constexpr std::array<u8, 2> ReturnProbePatch{0x0f, 0x0b};
        static constexpr std::array<u8, 4> FunctionPrologueExpected{0x55, 0x48, 0x89, 0xe5};
        static constexpr std::array<u8, 4> FunctionProloguePatch{0x0f, 0x0b, 0x90, 0x90};

        struct ExactPatch {
            uintptr_t guest_address;
            std::span<const u8> expected;
            std::span<const u8> replacement;
            std::string_view name;
        };
        const std::array exact_patches{
            ExactPatch{0x40bf01, GameUpdateCallExpected, GameUpdateCallPatch,
                       "LBP3 direct-level dispatch trigger"},
            ExactPatch{0x9444ba, ReturnProbeExpected, ReturnProbePatch,
                       "LBP3 direct-level return probe"},
            ExactPatch{0x9444c0, FunctionPrologueExpected, FunctionProloguePatch,
                       "LBP3 direct-level loader hook"},
            ExactPatch{0xc7c2b0, FunctionPrologueExpected, FunctionProloguePatch,
                       "LBP3 direct-level config hook"},
        };

        bool patches_valid = true;
        for (const auto& patch : exact_patches) {
            const uintptr_t image_offset = patch.guest_address - GuestImageBase;
            if (patch.guest_address < GuestImageBase ||
                image_offset + patch.expected.size() > g_eboot_image_size ||
                patch.expected.size() != patch.replacement.size() ||
                std::memcmp(reinterpret_cast<const void*>(g_eboot_address + image_offset),
                            patch.expected.data(), patch.expected.size()) != 0) {
                LOG_ERROR(Loader,
                          "Direct-level hook '{}' skipped: CUSA00063 01.26 code did not match at "
                          "{:#x}",
                          patch.name, patch.guest_address);
                patches_valid = false;
            }
        }
        if (patches_valid) {
            for (const auto& patch : exact_patches) {
                const uintptr_t image_offset = patch.guest_address - GuestImageBase;
                std::memcpy(reinterpret_cast<void*>(g_eboot_address + image_offset),
                            patch.replacement.data(), patch.replacement.size());
                LOG_INFO(Loader, "Applied {} at {:#x}", patch.name, patch.guest_address);
            }
        }
#else
        LOG_WARNING(Loader,
                    "LBP3 direct-level loading is currently supported only by the x86_64 macOS "
                    "build");
#endif
    }

    LOG_INFO(Loader,
             "LBP3 compatibility profile: prize_bubbles={}, disable_sprite_lights={}, "
             "disable_tone_map={}, direct_level={}",
             g_lbp3_patch_prize_bubbles, g_lbp3_disable_sprite_lights, g_lbp3_disable_tone_map,
             GetLbp3DirectLevelSpec());
}

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    bool versionMatches = metadataAppVer == app_version;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            try {
                                patchValue = convertValueToHex(type, patchValue);
                            } catch (std::exception& e) {
                                ASSERT_MSG(false,
                                           "Failed to parse patch value \"{}\" for \"{}\" in "
                                           "patch \"{}\", error: \"{}\"\n"
                                           "If the patch was working on earlier versions, then it "
                                           "was using a format that shadPS4 handled incorrectly, "
                                           "and the patch should instead be fixed.",
                                           patchValue, address, currentPatchName, e.what());
                            }
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        const patchInfo patch = {
                            .gameSerial = "*",
                            .modNameStr = currentPatchName,
                            .offsetStr = address,
                            .valueStr = patchValue,
                            .targetStr = targetStr,
                            .sizeStr = sizeStr,
                            .littleEndian = littleEndian,
                            .patchMask = patchMask,
                            .maskOffset = maskOffsetValue,
                        };
                        MemoryPatcher::PatchMemory(patch);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

void OnGameLoaded() {
    if (Core::PerfTelemetry::IsStartRequested()) {
        Core::PerfTelemetry::Start();
    }
    ApplyBuiltInLbp3CompatibilityPatches();
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (EmulatorState::GetInstance()->IsAutoPatchesLoadEnabled()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }
            std::ifstream json_file{repo.path() / "files.json"};
            nlohmann::json available_patches = nlohmann::json::parse(json_file);
            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
}

void AddPatchToQueue(const patchInfo& patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch);
    }

    pending_patches.clear();
}

void PatchMemory(const patchInfo& patch) {
    // Send a request to modify the process memory.
    void* cheatAddress = nullptr;

    if (patch.patchMask == PatchMask::None) {
        if (patch.isOffset) {
            cheatAddress =
                reinterpret_cast<void*>(g_eboot_address + std::stoi(patch.offsetStr, 0, 16));
        } else {
            cheatAddress = reinterpret_cast<void*>(g_eboot_address +
                                                   (std::stoi(patch.offsetStr, 0, 16) - 0x400000));
        }
    }

    if (patch.patchMask == PatchMask::Mask) {
        const uintptr_t match = PatternScan(patch.offsetStr);
        if (match == 0) {
            LOG_WARNING(Loader, "Patch mask not found: {}", patch.modNameStr);
            return;
        }
        cheatAddress = reinterpret_cast<void*>(match + patch.maskOffset);
    }

    if (patch.patchMask == PatchMask::Mask_Jump32) {
        int jumpSize = std::stoi(patch.sizeStr);

        constexpr int MAX_PATTERN_LENGTH = 256;
        if (jumpSize < 5) {
            LOG_ERROR(Loader, "Jump size must be at least 5 bytes");
            return;
        }
        if (jumpSize > MAX_PATTERN_LENGTH) {
            LOG_ERROR(Loader, "Jump size must be no more than {} bytes.", MAX_PATTERN_LENGTH);
            return;
        }

        // Find the base address using "Address"
        uintptr_t baseAddress = PatternScan(patch.offsetStr);
        if (baseAddress == 0) {
            LOG_ERROR(Loader, "PatternScan failed for mask_jump32 with pattern: {}",
                      patch.offsetStr);
            return;
        }
        uintptr_t patchAddress = baseAddress + patch.maskOffset;

        // Fills the original region (jumpSize bytes) with NOPs
        std::vector<u8> nopBytes(jumpSize, 0x90);
        std::memcpy(reinterpret_cast<void*>(patchAddress), nopBytes.data(), nopBytes.size());

        // Use "Target" to locate the start of the code cave
        uintptr_t jump_target = PatternScan(patch.targetStr);
        if (jump_target == 0) {
            LOG_ERROR(Loader, "PatternScan failed to Target with pattern: {}", patch.targetStr);
            return;
        }

        // Converts the Value attribute to a byte array (payload)
        std::vector<u8> payload;
        for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
            std::string tempStr = patch.valueStr.substr(i, 2);
            const char* byteStr = tempStr.c_str();
            char* endPtr;
            unsigned int byteVal = std::strtoul(byteStr, &endPtr, 16);

            if (endPtr != byteStr + 2) {
                LOG_ERROR(Loader, "Invalid byte in Value: {}", patch.valueStr.substr(i, 2));
                return;
            }
            payload.push_back(static_cast<u8>(byteVal));
        }

        // Calculates the end of the code cave (where the return jump will be inserted)
        uintptr_t code_cave_end = jump_target + payload.size();

        // Write the payload to the code cave, from jump_target
        std::memcpy(reinterpret_cast<void*>(jump_target), payload.data(), payload.size());

        // Inserts the initial jump in the original region to divert to the code cave
        u8 jumpInstruction[5];
        jumpInstruction[0] = 0xE9;
        s32 relJump = static_cast<s32>(jump_target - patchAddress - 5);
        std::memcpy(&jumpInstruction[1], &relJump, sizeof(relJump));
        std::memcpy(reinterpret_cast<void*>(patchAddress), jumpInstruction,
                    sizeof(jumpInstruction));

        // Inserts jump back at the end of the code cave to resume execution after patching
        u8 jumpBack[5];
        jumpBack[0] = 0xE9;
        // Calculates the relative offset to return to the instruction immediately following the
        // overwritten region
        s32 target_return = static_cast<s32>((patchAddress + jumpSize) - (code_cave_end + 5));
        std::memcpy(&jumpBack[1], &target_return, sizeof(target_return));
        std::memcpy(reinterpret_cast<void*>(code_cave_end), jumpBack, sizeof(jumpBack));

        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 patch.modNameStr, patchAddress, jump_target, code_cave_end, jumpSize);
        return;
    }

    if (cheatAddress == nullptr) {
        LOG_ERROR(Loader, "Failed to get address for patch {}", patch.modNameStr);
        return;
    }

    std::vector<unsigned char> bytePatch;

    for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
        unsigned char byte = static_cast<unsigned char>(
            std::strtol(patch.valueStr.substr(i, 2).c_str(), nullptr, 16));

        bytePatch.push_back(byte);
    }

    if (patch.littleEndian) {
        std::reverse(bytePatch.begin(), bytePatch.end());
    }

    std::memcpy(cheatAddress, bytePatch.data(), bytePatch.size());

    LOG_INFO(Loader, "Applied patch: {}, Offset: {:#x}, Value: {}", patch.modNameStr,
             (uintptr_t)cheatAddress, patch.valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    const char* start = pattern.data();
    const char* end = start + pattern.size();

    for (const char* current = start; current < end; ++current) {
        if (*current == '?') {
            ++current;
            if (*current == '?')
                ++current;
            bytes.push_back(-1);
        } else {
            bytes.push_back(strtoul(current, const_cast<char**>(&current), 16));
        }
    }

    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    std::vector<int32_t> patternBytes = PatternToByte(signature);
    const auto scanBytes = static_cast<uint8_t*>((void*)g_eboot_address);

    const int32_t* sigPtr = patternBytes.data();
    const size_t sigSize = patternBytes.size();

    if (sigSize == 0 || g_eboot_image_size < sigSize) {
        return 0;
    }

    uint32_t foundResults = 0;
    for (uint32_t i = 0; i < g_eboot_image_size - sigSize; ++i) {
        bool found = true;
        for (uint32_t j = 0; j < sigSize; ++j) {
            if (scanBytes[i + j] != sigPtr[j] && sigPtr[j] != -1) {
                found = false;
                break;
            }
        }

        if (found) {
            foundResults++;
            return reinterpret_cast<uintptr_t>(&scanBytes[i]);
        }
    }

    return 0;
}

} // namespace MemoryPatcher
