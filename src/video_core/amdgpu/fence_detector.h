// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/amdgpu/pm4_cmds.h"

namespace AmdGpu {

// Finds label writes which remain visible to the CPU. Label writes consumed by
// a later WAIT_REG_MEM in the same submission are GPU-internal synchronization
// and must not force buffer readback/protection work.
class FenceDetector {
public:
    explicit FenceDetector(std::span<const u32> commands, bool enabled) {
        if (enabled) {
            DetectFences(commands);
        }
    }

    [[nodiscard]] bool IsFence(const PM4Header* header) const {
        return std::ranges::contains(fences, header, &LabelWrite::packet);
    }

private:
    struct LabelWrite {
        const PM4Header* packet{};
        VAddr label{};
        u64 value{};
    };

    void DetectFences(std::span<const u32> commands) {
        while (!commands.empty()) {
            const auto* header = reinterpret_cast<const PM4Header*>(commands.data());
            switch (header->type.Value()) {
            case 0:
                // Type-0 packets are not expected in a DCB/ACB parsed here.
                return;
            case 2:
                commands = commands.subspan(1);
                continue;
            case 3:
                break;
            default:
                UNREACHABLE_MSG("Wrong PM4 type {}", header->type.Value());
            }

            const size_t packet_words = header->type3.NumWords() + 1;
            // Compute rings can end in a packet split across the ring boundary.
            // Liverpool buffers that packet for the next submission; do not
            // inspect fields which are not present in this span.
            if (packet_words > commands.size()) {
                return;
            }

            switch (header->type3.opcode.Value()) {
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (event_eos->command == PM4CmdEventWriteEos::Command::SignalFence) {
                    fences.push_back({header, reinterpret_cast<VAddr>(event_eos->Address<void*>()),
                                      event_eos->DataDWord()});
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (SelectsInterrupt(event_eop->int_sel)) {
                    fences.push_back({header});
                }
                if (event_eop->data_sel == DataSelect::Data32Low) {
                    fences.push_back({header, reinterpret_cast<VAddr>(event_eop->Address<void>()),
                                      event_eop->DataDWord()});
                } else if (event_eop->data_sel == DataSelect::Data64) {
                    fences.push_back({header, reinterpret_cast<VAddr>(event_eop->Address<void>()),
                                      event_eop->DataQWord()});
                }
                break;
            }
            case PM4ItOpcode::ReleaseMem: {
                const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
                if (release_mem->data_sel == DataSelect::Data32Low) {
                    fences.push_back({header,
                                      reinterpret_cast<VAddr>(release_mem->Address<void*>()),
                                      release_mem->DataDWord()});
                } else if (release_mem->data_sel == DataSelect::Data64) {
                    fences.push_back({header,
                                      reinterpret_cast<VAddr>(release_mem->Address<void*>()),
                                      release_mem->DataQWord()});
                } else if (release_mem->data_sel == DataSelect::GpuClock64 ||
                           release_mem->data_sel == DataSelect::PerfCounter ||
                           release_mem->data_sel == DataSelect::GdsMemStore) {
                    fences.push_back({header});
                } else if (release_mem->data_sel == DataSelect::None &&
                           SelectsInterrupt(release_mem->int_sel)) {
                    fences.push_back({header});
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * sizeof(u32);
                if (data_size <= sizeof(u64) && write_data->wr_confirm.Value()) {
                    u64 value{};
                    std::memcpy(&value, write_data->data, data_size);
                    fences.push_back(
                        {header, reinterpret_cast<VAddr>(write_data->Address<void*>()), value});
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                if (wait_reg_mem->mem_space == PM4CmdWaitRegMem::MemSpace::Register) {
                    break;
                }
                const VAddr wait_addr = reinterpret_cast<VAddr>(wait_reg_mem->Address<void*>());
                const u32 mask = wait_reg_mem->mask;
                const u32 reference = wait_reg_mem->ref;
                using Function = PM4CmdWaitRegMem::Function;
                std::erase_if(fences, [&](const LabelWrite& write) {
                    if (write.label == 0 || wait_addr != write.label) {
                        return false;
                    }
                    const u32 value = static_cast<u32>(write.value);
                    switch (wait_reg_mem->function.Value()) {
                    case Function::LessThan:
                        return (value & mask) < reference;
                    case Function::LessThanEqual:
                        return (value & mask) <= reference;
                    case Function::Equal:
                        return (value & mask) == reference;
                    case Function::NotEqual:
                        return (value & mask) != reference;
                    case Function::GreaterThanEqual:
                        return (value & mask) >= reference;
                    case Function::GreaterThan:
                        return (value & mask) > reference;
                    default:
                        UNREACHABLE();
                    }
                });
                break;
            }
            default:
                break;
            }
            commands = commands.subspan(packet_words);
        }
    }

    std::vector<LabelWrite> fences;
};

} // namespace AmdGpu
