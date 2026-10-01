// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include "common/types.h"

namespace VideoCore {

constexpr u64 QuadListIndexCount(u32 vertex_count) {
    return u64{vertex_count / 4} * 6;
}

// Expand backwards so source and destination may share one upload allocation.
// The 0--2 diagonal and winding match the existing non-indexed QuadList lowering.
// Vertex 0 remains last in both triangles for the last-vertex provoking convention.
// An incomplete final quad does not produce a primitive.
template <typename Index>
u32 ExpandQuadIndicesInPlace(std::span<Index> storage, u32 source_count,
                             std::optional<Index> restart_index = std::nullopt) {
    static_assert(std::is_same_v<Index, u16> || std::is_same_v<Index, u32>);
    const u64 output_count = QuadListIndexCount(source_count);
    if (output_count > storage.size() || output_count > std::numeric_limits<u32>::max()) {
        return 0;
    }
    // Games may leave restart enabled for every topology. Only an actual marker
    // changes quad assembly. Preserve the tessellation path if one occurs before
    // the incomplete tail; no destination bytes have been changed in that case.
    if (restart_index) {
        const auto source = storage.first((source_count / 4) * 4);
        if (std::ranges::find(source, *restart_index) != source.end()) {
            return 0;
        }
    }
    for (u32 quad = source_count / 4; quad > 0; --quad) {
        const u32 src = (quad - 1) * 4;
        const std::array<Index, 4> vertices{storage[src], storage[src + 1], storage[src + 2],
                                            storage[src + 3]};
        const u32 dst = (quad - 1) * 6;
        storage[dst] = vertices[1];
        storage[dst + 1] = vertices[2];
        storage[dst + 2] = vertices[0];
        storage[dst + 3] = vertices[2];
        storage[dst + 4] = vertices[3];
        storage[dst + 5] = vertices[0];
    }
    return static_cast<u32>(output_count);
}

} // namespace VideoCore
