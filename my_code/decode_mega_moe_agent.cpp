// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Decode raw device ATT after the workload exits. Retain instruction PC pairs;
// llvm-objdump supplies the ISA mapping without perturbing the application.
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {
rocprofiler_thread_trace_decoder_id_t decoder{};
std::filesystem::path output;
int64_t shader_engine = -1;
uint64_t wave_count = 0, info_count = 0;

void check(rocprofiler_status_t result, const char* name) {
    if (result != ROCPROFILER_STATUS_SUCCESS) {
        std::fprintf(stderr, "[decode] %s: %s\n", name, rocprofiler_get_status_string(result));
        std::exit(2);
    }
}

std::vector<char> read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { std::fprintf(stderr, "[decode] cannot read %s\n", path.c_str()); std::exit(2); }
    return {std::istreambuf_iterator<char>(stream), {}};
}

void records(rocprofiler_thread_trace_decoder_record_type_t type, void* data,
              uint64_t count, void*) {
    if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE) {
        auto* waves = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(data);
        for (uint64_t w = 0; w < count; ++w) {
            const auto& wave = waves[w];
            const auto name = "se" + std::to_string(shader_engine) + "_wave" +
                              std::to_string(++wave_count) + ".json";
            std::ofstream out(output / name);
            out << "{\"se\":" << shader_engine << ",\"cu\":" << unsigned(wave.cu)
                << ",\"simd\":" << unsigned(wave.simd) << ",\"slot\":" << unsigned(wave.wave_id)
                << ",\"wg\":" << unsigned(wave.workgroup_id) << ",\"cluster\":"
                << unsigned(wave.cluster_id) << ",\"contexts\":" << unsigned(wave.contexts)
                << ",\"begin\":" << wave.begin_time << ",\"end\":" << wave.end_time
                << ",\"instructions\":[";
            for (uint64_t i = 0; i < wave.instructions_size; ++i) {
                const auto& inst = wave.instructions_array[i];
                if (i) out << ',';
                out << '[' << inst.time << ',' << inst.category << ',' << inst.stall << ','
                    << inst.duration << ',' << inst.pc.address << ',' << inst.pc.code_object_id << ']';
            }
            out << "],\"timeline\":[";
            for (uint64_t i = 0; i < wave.timeline_size; ++i) {
                if (i) out << ',';
                out << '[' << wave.timeline_array[i].type << ',' << wave.timeline_array[i].duration << ']';
            }
            out << "]}\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY) {
        std::ofstream out(output / "occupancy.jsonl", std::ios::app);
        auto* events = static_cast<rocprofiler_thread_trace_decoder_occupancy_t*>(data);
        for (uint64_t i = 0; i < count; ++i) {
            const auto& e = events[i];
            out << "[" << shader_engine << ',' << e.time << ',' << unsigned(e.cu) << ','
                << unsigned(e.simd) << ',' << unsigned(e.wave_id) << ',' << e.start << ','
                << e.pc.address << ',' << e.pc.code_object_id << "]\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME) {
        std::ofstream out(output / "realtime.jsonl", std::ios::app);
        auto* events = static_cast<rocprofiler_thread_trace_decoder_realtime_t*>(data);
        for (uint64_t i = 0; i < count; ++i) {
            out << '[' << shader_engine << ',' << events[i].shader_clock << ','
                << events[i].realtime_clock << "]\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY) {
        std::ofstream out(output / "frequency.txt", std::ios::app);
        auto* values = static_cast<uint64_t*>(data);
        for (uint64_t i = 0; i < count; ++i) out << values[i] << '\n';
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO) {
        std::ofstream out(output / "info.jsonl", std::ios::app);
        auto* values = static_cast<rocprofiler_thread_trace_decoder_info_t*>(data);
        for (uint64_t i = 0; i < count; ++i) {
            ++info_count;
            const char* message = rocprofiler_thread_trace_decoder_info_string(decoder, values[i]);
            std::fprintf(stderr, "[decode] SE=%ld INFO=%d %s\n", shader_engine, int(values[i]),
                         message ? message : "");
            out << '[' << shader_engine << ',' << int(values[i]) << "]\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT) {
        std::ofstream out(output / "events.jsonl", std::ios::app);
        auto* events = static_cast<rocprofiler_thread_trace_decoder_event_t*>(data);
        for (uint64_t i = 0; i < count; ++i) {
            const auto& e = events[i];
            out << '[' << shader_engine << ',' << e.time << ',' << int(e.type) << ','
                << unsigned(e.me_id) << ',' << unsigned(e.pipe_id) << ',' << e.flags
                << ',' << e.payload.raw << ',' << e.byte_offset << "]\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH) {
        std::ofstream out(output / "dispatches.jsonl", std::ios::app);
        auto* events = static_cast<rocprofiler_thread_trace_decoder_dispatch_t*>(data);
        for (uint64_t i = 0; i < count; ++i) {
            const auto& e = events[i];
            out << '[' << shader_engine << ',' << e.time << ',' << unsigned(e.me_id)
                << ',' << unsigned(e.pipe_id) << ',' << e.flags << ',' << e.vgprs
                << ',' << e.sgprs << ',' << e.lds_size << ',' << e.thread_dim_x
                << ',' << e.thread_dim_y << ',' << e.thread_dim_z << ','
                << e.entry_point.address << ',' << e.entry_point.code_object_id
                << "]\n";
        }
    } else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP) {
        auto* values = static_cast<uint64_t*>(data);
        for (uint64_t i = 0; i < count; ++i) std::printf("GFXIP=%lu\n", values[i]);
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: decode_mega_moe_agent CODE_OBJECT_DIR OUTPUT_DIR DECODER_DIR [SE:RAW_ATT ...]\n");
        return 2;
    }
    const std::filesystem::path input = argv[1];
    output = argv[2];
    if (!std::filesystem::create_directory(output)) {
        std::fprintf(stderr, "[decode] output must be a new directory\n"); return 2;
    }
    check(rocprofiler_thread_trace_decoder_create(&decoder, argv[3]), "create decoder");
    std::ifstream index(input / "code_objects.tsv");
    if (!index) { std::fprintf(stderr, "[decode] missing code_objects.tsv\n"); return 2; }
    std::vector<std::vector<char>> retained;
    uint64_t id, size, bytes;
    int64_t delta;
    std::string name;
    while (index >> id >> delta >> size >> bytes >> name) {
        retained.emplace_back(read(input / name));
        const auto& binary = retained.back();
        check(rocprofiler_thread_trace_decoder_codeobj_load(
            decoder, id, static_cast<uint64_t>(delta), size, binary.data(), binary.size()),
            "load code object");
    }
    for (int i = 4; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto separator = argument.find(':');
        shader_engine = std::stoll(argument.substr(0, separator));
        auto binary = read(argument.substr(separator + 1));
        if (binary.empty()) { std::fprintf(stderr, "[decode] empty ATT\n"); return 2; }
        std::printf("DECODING SE=%ld BYTES=%zu\n", shader_engine, binary.size());
        check(rocprofiler_trace_decode(decoder, records, binary.data(), binary.size(), nullptr),
              "decode ATT");
    }
    rocprofiler_thread_trace_decoder_destroy(decoder);
    std::printf("DECODED_WAVES=%lu INFO_RECORDS=%lu\n", wave_count, info_count);
    return wave_count ? 0 : 3;
}
