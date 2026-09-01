#include <ATen/cuda/CUDAContext.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t err = (call);                                                  \
    TORCH_CHECK(err == hipSuccess, #call " failed: ", hipGetErrorString(err)); \
  } while (0)

namespace {

constexpr char kKernelSymbol[] =
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4";
constexpr char kBuildKey[] = "@MOE_CPP_EXTENSION_BUILD_KEY@";
constexpr char kCodeObjectSha256[] = "@MOE_CODE_OBJECT_SHA256@";

constexpr std::size_t kKernargSize = 184;
constexpr std::int64_t kTileM = 64;
constexpr std::int64_t kTileN = 256;
constexpr std::int64_t kBakedK = 7168;
constexpr std::int64_t kExperts = 96;
constexpr std::int64_t kValidRowsPerExpert = 32;
constexpr std::int64_t kTopK = 6;

struct ModuleState {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
};

std::mutex g_module_mutex;
std::unordered_map<std::string, ModuleState> g_modules;

template <typename T>
T read_kernarg(const std::string& payload, std::size_t offset) {
  TORCH_CHECK(
      offset + sizeof(T) <= payload.size(),
      "kernarg read at ",
      offset,
      " exceeds ",
      payload.size(),
      " bytes");
  T value{};
  std::memcpy(&value, payload.data() + offset, sizeof(value));
  return value;
}

void check_tensor(
    const char* name,
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    int device_index) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/ROCm tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
  TORCH_CHECK(
      tensor.get_device() == device_index,
      name,
      " must be on cuda:",
      device_index);
}

void check_pointer(
    const std::string& payload,
    std::size_t offset,
    const char* name,
    const torch::Tensor& tensor) {
  const auto packed = read_kernarg<std::uint64_t>(payload, offset);
  const auto actual =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()));
  TORCH_CHECK(
      packed == actual,
      name,
      " pointer in the 184-byte kernarg differs from tensor.data_ptr()");
}

unsigned int checked_dim(std::int64_t value, const char* name) {
  TORCH_CHECK(
      value > 0 &&
          static_cast<std::uint64_t>(value) <=
              std::numeric_limits<unsigned int>::max(),
      name,
      " does not fit a positive HIP launch dimension: ",
      value);
  return static_cast<unsigned int>(value);
}

hipFunction_t load_kernel(const std::string& code_object_path, int device_index) {
  const std::string key =
      std::to_string(device_index) + ":" + code_object_path;
  std::lock_guard<std::mutex> guard(g_module_mutex);
  auto found = g_modules.find(key);
  if (found != g_modules.end()) {
    return found->second.function;
  }

  ModuleState state;
  HIP_CHECK(hipModuleLoad(&state.module, code_object_path.c_str()));
  hipError_t get_function =
      hipModuleGetFunction(&state.function, state.module, kKernelSymbol);
  if (get_function != hipSuccess) {
    (void)hipModuleUnload(state.module);
    TORCH_CHECK(
        false,
        "hipModuleGetFunction(",
        kKernelSymbol,
        ") failed: ",
        hipGetErrorString(get_function));
  }
  auto inserted = g_modules.emplace(key, state);
  return inserted.first->second.function;
}

torch::Tensor launch_moe_gemm1(
    torch::Tensor out,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor m_tile_map,
    torch::Tensor bias,
    torch::Tensor quant_scale,
    py::bytes kernarg_bytes,
    const std::vector<std::int64_t>& grid,
    const std::vector<std::int64_t>& block,
    const std::vector<std::int64_t>& cluster,
    std::int64_t m_per_expert,
    std::int64_t n,
    std::int64_t k,
    std::int64_t experts,
    const std::string& code_object_path) {
  RECORD_FUNCTION(
      "moe_gemm1_cpp_launcher::launch",
      std::vector<c10::IValue>());
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA/ROCm tensor");
  const int device_index = out.get_device();
  const c10::cuda::OptionalCUDAGuard device_guard(device_index);
  HIP_CHECK(hipSetDevice(device_index));

  check_tensor("out", out, at::kBFloat16, device_index);
  check_tensor("a", a, at::kByte, device_index);
  check_tensor("b", b, at::kByte, device_index);
  check_tensor("scale_a", scale_a, at::kByte, device_index);
  check_tensor("scale_b", scale_b, at::kByte, device_index);
  check_tensor("m_tile_map", m_tile_map, at::kInt, device_index);
  const bool bias_aliases_a = bias.data_ptr() == a.data_ptr();
  const bool quant_scale_aliases_out =
      quant_scale.data_ptr() == out.data_ptr();
  TORCH_CHECK(
      bias_aliases_a == quant_scale_aliases_out,
      "bias and quant_scale must use either both standalone tensors or both "
      "pipeline aliases");
  const bool pipeline_contract = bias_aliases_a;
  if (pipeline_contract) {
    check_tensor("bias alias", bias, at::kByte, device_index);
    check_tensor(
        "quant_scale alias", quant_scale, at::kBFloat16, device_index);
  } else {
    check_tensor("bias", bias, at::kFloat, device_index);
    check_tensor("quant_scale", quant_scale, at::kFloat, device_index);
  }

  TORCH_CHECK(m_per_expert == kTileM, "this launcher requires M=64");
  TORCH_CHECK(n > 0 && n % kTileN == 0, "N must be a positive multiple of 256");
  TORCH_CHECK(k == kBakedK, "this launcher requires K=7168");
  TORCH_CHECK(experts == kExperts, "this launcher requires 96 experts");

  const std::string payload = kernarg_bytes;
  TORCH_CHECK(
      payload.size() == kKernargSize,
      "kernarg must be exactly 184 bytes, got ",
      payload.size());
  const auto contiguous_m =
      static_cast<std::int64_t>(read_kernarg<std::uint32_t>(payload, 164));
  const auto packed_n =
      static_cast<std::int64_t>(read_kernarg<std::uint32_t>(payload, 168));
  const std::int64_t routed_rows = experts * kValidRowsPerExpert;
  const std::int64_t upper_bound =
      routed_rows + experts * kTileM - kTopK;
  const std::int64_t expected_contiguous_m =
      ((upper_bound + kTileM - 1) / kTileM) * kTileM;
  TORCH_CHECK(
      contiguous_m == expected_contiguous_m,
      "kernarg i32_m is ",
      contiguous_m,
      ", expected ",
      expected_contiguous_m);
  TORCH_CHECK(packed_n == n, "kernarg i32_n does not match N");

  TORCH_CHECK(
      out.numel() == contiguous_m * (n / 2),
      "out footprint does not match [contiguous_m, N/2]");
  TORCH_CHECK(
      a.numel() == contiguous_m * (k / 2),
      "A footprint does not match packed MXFP4 activation");
  TORCH_CHECK(
      b.numel() == experts * n * (k / 2),
      "B footprint does not match packed expert weights");
  TORCH_CHECK(
      scale_a.numel() == contiguous_m * (k / 32),
      "scale_a footprint does not match e8m0 blocks");
  TORCH_CHECK(
      scale_b.numel() == experts * n * (k / 32),
      "scale_b footprint does not match expert e8m0 blocks");
  TORCH_CHECK(m_tile_map.numel() == experts, "m_tile_map must have 96 entries");
  if (pipeline_contract) {
    TORCH_CHECK(
        bias.numel() == a.numel(),
        "pipeline bias alias must cover the same storage view as A");
    TORCH_CHECK(
        quant_scale.numel() == out.numel(),
        "pipeline quant_scale alias must cover the same storage view as out");
  } else {
    TORCH_CHECK(bias.numel() == n, "bias footprint must be N");
    TORCH_CHECK(quant_scale.numel() == 1, "quant_scale must have one element");
  }

  check_pointer(payload, 0, "out", out);
  check_pointer(payload, 40, "a", a);
  check_pointer(payload, 48, "b", b);
  check_pointer(payload, 56, "scale_a", scale_a);
  check_pointer(payload, 96, "scale_b", scale_b);
  check_pointer(payload, 112, "m_tile_map", m_tile_map);
  check_pointer(payload, 120, "bias", bias);
  check_pointer(payload, 128, "quant_scale", quant_scale);

  TORCH_CHECK(read_kernarg<std::uint32_t>(payload, 8) == 1, "C size0 mismatch");
  TORCH_CHECK(
      read_kernarg<std::uint32_t>(payload, 12) == contiguous_m,
      "C size1 mismatch");
  TORCH_CHECK(
      read_kernarg<std::uint32_t>(payload, 16) == n / 2,
      "C size2 mismatch");
  TORCH_CHECK(
      read_kernarg<std::uint64_t>(payload, 20) ==
          static_cast<std::uint64_t>(contiguous_m * (n / 2)),
      "C stride0 mismatch");
  TORCH_CHECK(
      read_kernarg<std::uint64_t>(payload, 28) ==
          static_cast<std::uint64_t>(n / 2),
      "C stride1 mismatch");
  if (pipeline_contract) {
    const std::int64_t sa_size1 = contiguous_m / 4;
    const std::int64_t sa_size2 = k / 32;
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 64) == 1,
        "pipeline scale_a size0 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 68) == sa_size1,
        "pipeline scale_a size1 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 72) == sa_size2,
        "pipeline scale_a size2 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint64_t>(payload, 76) ==
            static_cast<std::uint64_t>(sa_size1 * sa_size2),
        "pipeline scale_a stride0 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint64_t>(payload, 84) ==
            static_cast<std::uint64_t>(sa_size2),
        "pipeline scale_a stride1 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 104) ==
            static_cast<std::uint32_t>(scale_b.numel() / sizeof(std::uint32_t)),
        "pipeline scale_b size0 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 136) == 1,
        "pipeline quant_scale size0 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 140) == contiguous_m,
        "pipeline quant_scale size1 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 144) == n / 2,
        "pipeline quant_scale size2 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint64_t>(payload, 148) ==
            static_cast<std::uint64_t>(contiguous_m * (n / 2)),
        "pipeline quant_scale stride0 mismatch");
    TORCH_CHECK(
        read_kernarg<std::uint64_t>(payload, 156) ==
            static_cast<std::uint64_t>(n / 2),
        "pipeline quant_scale stride1 mismatch");
    TORCH_CHECK(
        read_kernarg<float>(payload, 172) == 7.0f,
        "pipeline swiglu_limit mismatch");
    TORCH_CHECK(
        read_kernarg<float>(payload, 176) == 4.0f,
        "pipeline situ_beta mismatch");
    TORCH_CHECK(
        read_kernarg<float>(payload, 180) == 25.0f,
        "pipeline situ_linear_beta mismatch");
  } else {
    TORCH_CHECK(
        read_kernarg<std::uint32_t>(payload, 104) == experts,
        "standalone scale_b size0 mismatch");
  }

  TORCH_CHECK(grid.size() == 3, "grid must have three dimensions");
  TORCH_CHECK(block.size() == 3, "block must have three dimensions");
  TORCH_CHECK(cluster.size() == 3, "cluster must have three dimensions");
  const std::int64_t expected_grid_x =
      (contiguous_m / kTileM) * (n / kTileN);
  TORCH_CHECK(
      grid[0] == expected_grid_x && grid[1] == 1 && grid[2] == 1,
      "grid must match the MoE runner geometry");
  TORCH_CHECK(
      block[0] == 128 && block[1] == 1 && block[2] == 1,
      "block must be (128,1,1)");
  TORCH_CHECK(
      cluster[0] == 1 && cluster[1] == 1 && cluster[2] == 1,
      "cluster must be (1,1,1)");

  hipFunction_t function = load_kernel(code_object_path, device_index);
  hipStream_t stream =
      at::cuda::getCurrentCUDAStream(device_index).stream();
  std::size_t arg_size = payload.size();
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      const_cast<char*>(payload.data()),
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &arg_size,
      HIP_LAUNCH_PARAM_END};

  // This exact kernel's audited cluster is (1,1,1), so the ordinary module
  // API represents identical geometry without dropping a nontrivial cluster
  // attribute.  hipDrvLaunchKernelEx is required only when a cluster
  // dimension exceeds one.
  HIP_CHECK(hipModuleLaunchKernel(
      function,
      checked_dim(grid[0], "grid.x"),
      checked_dim(grid[1], "grid.y"),
      checked_dim(grid[2], "grid.z"),
      checked_dim(block[0], "block.x"),
      checked_dim(block[1], "block.y"),
      checked_dim(block[2], "block.z"),
      0,
      stream,
      nullptr,
      reinterpret_cast<void**>(&extra)));
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "launch",
      &launch_moe_gemm1,
      "Launch the exact moe_gemm1_a4w4_v0.s code object",
      py::arg("out"),
      py::arg("a"),
      py::arg("b"),
      py::arg("scale_a"),
      py::arg("scale_b"),
      py::arg("m_tile_map"),
      py::arg("bias"),
      py::arg("quant_scale"),
      py::arg("kernarg"),
      py::arg("grid"),
      py::arg("block"),
      py::arg("cluster"),
      py::arg("m_per_expert"),
      py::arg("n"),
      py::arg("k"),
      py::arg("experts"),
      py::arg("code_object_path"));
  m.def("build_key", []() { return std::string(kBuildKey); });
  m.def(
      "code_object_sha256",
      []() { return std::string(kCodeObjectSha256); });
  m.def("kernel_symbol", []() { return std::string(kKernelSymbol); });
}
