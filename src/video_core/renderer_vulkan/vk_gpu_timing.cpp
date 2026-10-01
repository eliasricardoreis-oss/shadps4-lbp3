// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_gpu_timing.h"
#include "video_core/renderer_vulkan/vk_instance.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

namespace Vulkan {
namespace {
using Clock = std::chrono::steady_clock;
s64 Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}
const char* KindName(GpuTiming::Kind kind) {
    switch (kind) {
    case GpuTiming::Kind::Detile:
        return "detile_dispatch";
    case GpuTiming::Kind::Tile:
        return "tile_download_dispatch";
    case GpuTiming::Kind::Render:
        return "render_pass";
    default:
        return "submission";
    }
}
} // namespace

struct GpuTiming::Impl {
    struct Scope {
        Work work{};
        bool ended{};
        bool truncated{};
    };
    struct Batch {
        u64 tick{};
        s64 begin_ns{}, submit_ns{};
        u32 count{}, render_passes{}, overflow{}, render_phase{};
        std::array<Scope, MaxScopes> scopes{};
        // Query 0/1 bracket the submission; each scope reserves a start/end pair.
        std::array<u64, 2 * QueryCount> values{};
    };
    struct Bank {
        vk::UniqueQueryPool pool;
        Batch batch;
        bool pending{};
        u32 retries{};
    };

    vk::Device device;
    std::filesystem::path directory, stem;
    const char* role;
    u32 period, sequence{}, valid_bits{};
    size_t file_limit;
    double timestamp_period{};
    u64 mask{};
    std::array<Bank, BankCount> banks;
    Bank* active{};
    Token render_token{};
    std::ofstream csv;
    std::atomic_bool accepting{true};
    std::atomic<u64> skipped_banks{}, failed_queries{}, dropped_batches{}, dropped_limit{};
    std::atomic<u64> sampled{}, truncated{}, scope_overflow{}, incomplete{};
    u64 written_batches{}, written_rows{}, written_bytes{};
    std::array<Batch, BankCount> queue;
    u32 queued{}, head{}, tail{};
    std::mutex mutex;
    std::condition_variable_any wake;
    std::jthread writer;

    Impl(const Instance& instance, const char* role_, u32 period_, size_t limit, const char* path)
        : device{instance.GetDevice()}, directory{path}, role{role_}, period{period_},
          file_limit{std::clamp(limit, size_t{512}, FileLimit)} {
        auto families = instance.GetPhysicalDevice().getQueueFamilyProperties();
        valid_bits = families.at(instance.GetGraphicsQueueFamilyIndex()).timestampValidBits;
        timestamp_period = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
        if (!valid_bits || timestamp_period <= 0 || !period) {
            accepting = false;
            return;
        }
        mask = valid_bits == 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            accepting = false;
            return;
        }
        stem = directory / (std::string{role} + "_" + std::to_string(Now()));
        csv.open(stem.string() + ".csv");
        if (!csv) {
            accepting = false;
            return;
        }
        const std::string header =
            "scheduler,tick,record_begin_steady_ns,submit_steady_ns,kind,index,status,"
            "gpu_begin_ticks,gpu_end_ticks,gpu_ms,offset_ms,render_passes,scope_count,"
            "scope_overflow,address,bytes,width,height,depth,pitch,bits,tile_mode,mips,layers,"
            "render_pass,draw_calls,indirect_calls,shader_changes,vertices,first_vs,first_ps,last_"
            "vs,last_ps\n";
        csv << header;
        written_bytes = header.size();
        Status();
        writer = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    ~Impl() {
        accepting = false;
        for (const auto& bank : banks) {
            incomplete += bank.pending || &bank == active;
        }
        if (writer.joinable()) {
            writer.request_stop();
            wake.notify_one();
            writer.join();
        }
        Status();
    }

    void Status() {
        if (stem.empty())
            return;
        std::ofstream out{stem.string() + "_status.txt"};
        out << "version=2\nscheduler=" << role << "\nsample_period=" << period
            << "\nrender_sample_stride=" << RenderSampleStride
            << "\ntimestamp_period_ns=" << std::setprecision(12) << timestamp_period
            << "\ntimestamp_valid_bits=" << valid_bits << "\nfile_limit_bytes=" << file_limit
            << "\nsampled_submissions=" << sampled << "\nwritten_batches=" << written_batches
            << "\nwritten_rows=" << written_rows << "\nwritten_bytes=" << written_bytes
            << "\nskipped_banks=" << skipped_banks << "\nfailed_queries=" << failed_queries
            << "\ndropped_batches=" << dropped_batches << "\ndropped_limit=" << dropped_limit
            << "\ntruncated_scopes=" << truncated << "\nscope_overflow=" << scope_overflow
            << "\nincomplete_banks_at_close=" << incomplete << "\nio_ok=" << bool(csv) << '\n';
    }

    void Write(const Batch& batch) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        const auto& v = batch.values;
        const u64 span = (v[2] - v[0]) & mask;
        for (u32 i = 0; i <= batch.count; ++i) {
            const auto scope = i ? batch.scopes[i - 1] : Scope{};
            const auto& w = scope.work;
            const u32 q = 2 * i;
            const u64 start = v[2 * q] & mask, end = v[2 * q + 2] & mask;
            const u64 delta = (end - start) & mask, offset = (start - v[0]) & mask;
            const char* status = "ok";
            if (!v[2 * q + 1] || !v[2 * q + 3])
                status = "unavailable";
            else if (scope.truncated)
                status = "truncated";
            else if (offset > span || delta > span - offset || delta * timestamp_period > 60e9)
                status = "outside_submission";
            else if (!delta)
                status = "zero_interval";
            out << role << ',' << batch.tick << ',' << batch.begin_ns << ',' << batch.submit_ns
                << ',' << KindName(w.kind) << ',' << i << ',' << status << ',' << start << ','
                << end << ',' << delta * timestamp_period / 1e6 << ','
                << offset * timestamp_period / 1e6 << ',' << batch.render_passes << ','
                << batch.count << ',' << batch.overflow << ',' << w.address << ',' << w.bytes << ','
                << w.width << ',' << w.height << ',' << w.depth << ',' << w.pitch << ',' << w.bits
                << ',' << w.tile_mode << ',' << w.mips << ',' << w.layers << ',' << w.render_pass
                << ',' << w.draw_calls << ',' << w.indirect_calls << ',' << w.shader_changes << ','
                << w.vertices << ',' << w.first_vs << ',' << w.first_ps << ',' << w.last_vs << ','
                << w.last_ps << '\n';
        }
        const auto text = out.str();
        if (written_bytes + text.size() > file_limit) {
            ++dropped_limit;
            accepting = false;
            return;
        }
        csv << text;
        csv.flush();
        if (!csv) {
            accepting = false;
            ++dropped_batches;
            return;
        }
        written_bytes += text.size();
        ++written_batches;
        written_rows += batch.count + 1;
    }

    void Run(std::stop_token stop) {
        for (;;) {
            Batch batch;
            {
                std::unique_lock lock{mutex};
                wake.wait(lock, stop, [this] { return queued != 0; });
                if (!queued)
                    break;
                batch = queue[head];
                head = (head + 1) % BankCount;
                --queued;
            }
            Write(batch);
            Status();
        }
    }

    void Enqueue(const Batch& batch) {
        std::unique_lock lock{mutex, std::try_to_lock};
        if (!lock || queued == BankCount || !accepting) {
            ++dropped_batches;
            return;
        }
        queue[tail] = batch;
        tail = (tail + 1) % BankCount;
        ++queued;
        lock.unlock();
        wake.notify_one();
    }
};

GpuTiming::GpuTiming(const Instance& instance, const char* role, u32 period, size_t file_limit) {
    const char* directory = std::getenv("SHADPS4_GPU_TIMING_DIR");
    const char* precise = std::getenv("MESA_KK_PRECISE_COMPUTE_TIMESTAMPS");
    if (directory && *directory && precise && std::string_view{precise} == "1" &&
        instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp) {
        impl = std::make_unique<Impl>(instance, role, period, file_limit, directory);
    }
}
GpuTiming::~GpuTiming() = default;

void GpuTiming::Begin(vk::CommandBuffer command, u64 tick) {
    if (!impl || !impl->accepting || impl->sequence++ % impl->period)
        return;
    auto& p = *impl;
    std::error_code error;
    if (!std::filesystem::exists(p.directory / "enable", error) || error)
        return;
    for (auto& bank : p.banks) {
        if (bank.pending)
            continue;
        if (!bank.pool) {
            auto result = p.device.createQueryPoolUnique(
                {.queryType = vk::QueryType::eTimestamp, .queryCount = QueryCount});
            if (result.result != vk::Result::eSuccess) {
                ++p.failed_queries;
                p.accepting = false;
                return;
            }
            bank.pool = std::move(result.value);
        }
        bank.batch = {};
        bank.batch.tick = tick;
        bank.batch.begin_ns = Now();
        bank.batch.render_phase = p.sampled % RenderSampleStride;
        bank.retries = 0;
        p.active = &bank;
        command.resetQueryPool(*bank.pool, 0, QueryCount);
        command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool, 0);
        ++p.sampled;
        return;
    }
    ++p.skipped_banks;
}

void GpuTiming::End(vk::CommandBuffer command, u64 tick) {
    if (!impl || !impl->active)
        return;
    auto& p = *impl;
    auto& bank = *p.active;
    for (u32 i = 0; i < bank.batch.count; ++i) {
        auto& scope = bank.batch.scopes[i];
        if (!scope.ended) {
            command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool,
                                    3 + 2 * i);
            scope.ended = scope.truncated = true;
            ++p.truncated;
        }
    }
    command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool, 1);
    bank.batch.submit_ns = Now();
    bank.pending = true;
    p.active = nullptr;
    p.render_token = {};
}

void GpuTiming::Collect(u64 completed_tick) {
    if (!impl)
        return;
    auto& p = *impl;
    for (auto& bank : p.banks) {
        if (!bank.pending || bank.batch.tick > completed_tick)
            continue;
        auto& batch = bank.batch;
        const auto result = p.device.getQueryPoolResults(
            *bank.pool, 0, 2 + 2 * batch.count, sizeof(batch.values), batch.values.data(),
            2 * sizeof(u64),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (result == vk::Result::eNotReady && ++bank.retries < 3)
            continue;
        if (result == vk::Result::eSuccess)
            p.Enqueue(batch);
        else
            ++p.failed_queries;
        bank.pending = false;
    }
}

void GpuTiming::RenderPass() {
    if (impl && impl->active)
        ++impl->active->batch.render_passes;
}

void GpuTiming::BeginRenderPass(vk::CommandBuffer command, u32 width, u32 height, u32 layers) {
    if (!impl || !impl->active)
        return;
    auto& batch = impl->active->batch;
    const u32 ordinal = batch.render_passes++;
    impl->render_token = {};
    // Rotate the selected passes across sampled submissions. Do not add a render
    // boundary or a timestamp inside rendering. All work shares the bounded pool.
    if (ordinal % RenderSampleStride != batch.render_phase)
        return;
    impl->render_token = BeginWork(command, {.kind = Kind::Render,
                                             .width = width,
                                             .height = height,
                                             .layers = layers,
                                             .render_pass = ordinal + 1});
}

void GpuTiming::EndRenderPass(vk::CommandBuffer command) {
    if (!impl)
        return;
    EndWork(command, impl->render_token);
    impl->render_token = {};
}

bool GpuTiming::IsTimingRenderPass() const {
    return impl && impl->active && impl->render_token.tick &&
           impl->render_token.tick == impl->active->batch.tick;
}

void GpuTiming::RecordDraw(u64 vs, u64 ps, u64 vertices, bool indirect) {
    if (!IsTimingRenderPass())
        return;
    auto& w = impl->active->batch.scopes[impl->render_token.index].work;
    if (!w.draw_calls) {
        w.first_vs = vs;
        w.first_ps = ps;
    } else if (w.last_vs != vs || w.last_ps != ps) {
        ++w.shader_changes;
    }
    w.last_vs = vs;
    w.last_ps = ps;
    ++w.draw_calls;
    w.indirect_calls += indirect;
    w.vertices += vertices;
}

GpuTiming::Token GpuTiming::BeginWork(vk::CommandBuffer command, const Work& work) {
    if (!impl || !impl->active)
        return {};
    auto& bank = *impl->active;
    if (bank.batch.count == MaxScopes) {
        ++bank.batch.overflow;
        ++impl->scope_overflow;
        return {};
    }
    const u32 i = bank.batch.count++;
    bank.batch.scopes[i].work = work;
    command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool, 2 + 2 * i);
    return {bank.batch.tick, i};
}

void GpuTiming::EndWork(vk::CommandBuffer command, Token token) {
    if (!impl || !impl->active || !token.tick)
        return;
    auto& bank = *impl->active;
    if (token.tick != bank.batch.tick || token.index >= bank.batch.count ||
        bank.batch.scopes[token.index].ended)
        return;
    command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *bank.pool,
                            3 + 2 * token.index);
    bank.batch.scopes[token.index].ended = true;
}
} // namespace Vulkan
