// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Device-based ATT collection. No kernel-dispatch tracing/interception service
// is configured. The Python runner brackets the GEMM2 launch explicitly.
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
rocprofiler_context_id_t trace_context{}, code_context{};
std::filesystem::path output;
std::mutex output_mutex;
bool initialized = false;
bool active = false;
uint64_t capture_id = 0;

void check(rocprofiler_status_t status, const char* operation) {
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::fprintf(stderr, "[device ATT] %s: %s\n", operation,
                     rocprofiler_get_status_string(status));
        std::abort();
    }
}

uint64_t parameter(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    return value ? std::strtoull(value, nullptr, 0) : fallback;
}

void shader_data(rocprofiler_thread_trace_shader_data_t data,
                 rocprofiler_user_data_t) {
    std::lock_guard<std::mutex> lock(output_mutex);
    const std::string prefix = "capture" + std::to_string(capture_id) + "_se" +
                               std::to_string(data.shader_engine_id);
    std::ofstream raw(output / (prefix + ".att"), std::ios::binary | std::ios::app);
    const auto bytes = data.data_size - data.read_offset;
    raw.write(static_cast<const char*>(data.data) + data.read_offset, bytes);
    std::ofstream chunks(output / "chunks.jsonl", std::ios::app);
    chunks << "{\"capture\":" << capture_id << ",\"se\":" << data.shader_engine_id
           << ",\"chunk\":" << data.chunk_index << ",\"bytes\":" << bytes
           << ",\"read_offset\":" << data.read_offset << ",\"flags\":"
           << static_cast<uint64_t>(data.flags) << "}\n";
    std::fprintf(stderr, "[device ATT] capture=%lu SE=%ld chunk=%lu bytes=%lu flags=%lu\n",
                 capture_id, data.shader_engine_id, data.chunk_index, bytes,
                 static_cast<uint64_t>(data.flags));
}

void code_object(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*, void*) {
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.operation != ROCPROFILER_CODE_OBJECT_LOAD ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;
    const auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
        record.payload);
    std::vector<char> bytes;
    if (data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
        const auto* begin = reinterpret_cast<const char*>(data->memory_base);
        bytes.assign(begin, begin + data->memory_size);
    } else if (data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) {
        const std::string fd_path = "/proc/self/fd/" + std::to_string(data->storage_file);
        std::ifstream file(fd_path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(file), {});
    }
    if (bytes.empty()) {
        std::fprintf(stderr, "[device ATT] empty code object %lu: %s\n",
                     data->code_object_id, data->uri ? data->uri : "");
        return;
    }
    std::lock_guard<std::mutex> lock(output_mutex);
    const std::string file_name = "code_object_" + std::to_string(data->code_object_id) + ".out";
    std::ofstream file(output / file_name, std::ios::binary);
    file.write(bytes.data(), bytes.size());
    std::ofstream manifest(output / "code_objects.jsonl", std::ios::app);
    manifest << "{\"id\":" << data->code_object_id << ",\"load_delta\":"
             << data->load_delta << ",\"load_size\":" << data->load_size
             << ",\"bytes\":" << bytes.size() << ",\"file\":\"" << file_name
             << "\"}\n";
    std::ofstream index(output / "code_objects.tsv", std::ios::app);
    index << data->code_object_id << '\t' << data->load_delta << '\t' << data->load_size
          << '\t' << bytes.size() << '\t' << file_name << '\n';
}

rocprofiler_status_t agents(rocprofiler_agent_version_t, const void** values,
                            size_t count, void*) {
    const auto selected = parameter("AITER_ATT_GPU_INDEX", 0);
    bool found = false;
    for (size_t i = 0; i < count; ++i) {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(values[i]);
        if (agent->type != ROCPROFILER_AGENT_TYPE_GPU ||
            static_cast<uint64_t>(agent->logical_node_type_id) != selected) continue;
        rocprofiler_thread_trace_parameter_t parameters[] = {
            {ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU,
             {parameter("AITER_ATT_TARGET_CU", 1)}},
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK,
             {parameter("AITER_ATT_SE_MASK", 0xf)}},
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT,
             {parameter("AITER_ATT_SIMD", 0)}},
            {ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE,
             {parameter("AITER_ATT_BUFFER_BYTES", 0x10000000)}},
        };
        check(rocprofiler_configure_device_thread_trace_service(
            trace_context, agent->id, parameters, sizeof(parameters) / sizeof(parameters[0]),
            shader_data, {}), "configure device ATT");
        std::fprintf(stderr, "[device ATT] agent=%lu gpu_index=%d name=%s\n",
                     agent->id.handle, agent->logical_node_type_id, agent->name);
        found = true;
    }
    if (!found) {
        std::fprintf(stderr, "[device ATT] GPU index %lu not found\n", selected);
        std::abort();
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int initialize(rocprofiler_client_finalize_t, void*) {
    const char* directory = std::getenv("AITER_ATT_OUTPUT");
    if (!directory) std::abort();
    output = std::filesystem::path(directory) / ("pid" + std::to_string(getpid()));
    std::filesystem::create_directories(output);
    check(rocprofiler_create_context(&trace_context), "create device context");
    check(rocprofiler_create_context(&code_context), "create code object context");
    check(rocprofiler_configure_callback_tracing_service(
        code_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
        code_object, nullptr), "configure code object tracing");
    check(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, agents,
                                             sizeof(rocprofiler_agent_t), nullptr),
          "query agents");
    check(rocprofiler_start_context(code_context), "start code object context");
    initialized = true;
    std::fprintf(stderr, "[device ATT] initialized without dispatch interception\n");
    return 0;
}

void finalize(void*) {
    if (active) check(rocprofiler_stop_context(trace_context), "stop final device context");
    std::fprintf(stderr, "[device ATT] finalized captures=%lu\n", capture_id);
}
}  // namespace

extern "C" int aiter_att_start() {
    if (!initialized || active) return -1;
    ++capture_id;
    check(rocprofiler_start_context(trace_context), "start device context");
    active = true;
    return 0;
}

extern "C" int aiter_att_stop() {
    if (!initialized || !active) return -1;
    check(rocprofiler_stop_context(trace_context), "stop device context");
    active = false;
    return 0;
}

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id) {
    if (priority > 0) return nullptr;
    id->name = "AITER GEMM2 device ATT";
    static rocprofiler_tool_configure_result_t config{
        sizeof(rocprofiler_tool_configure_result_t), initialize, finalize, nullptr};
    return &config;
}
