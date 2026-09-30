// Standalone pybind11 binding for the NADPE MMA-Emu FP8 operator.
//
// Compiled together with the UNMODIFIED upstream source
//   micro26-ae/csrc/quantization/mma_emu/fp8_gemm_kernels.cu
// (Apache-2.0, NADPE / MICRO'26 "Not All Dot Products Are Equal").
// No vLLM headers are involved: fp8_gemm_kernels.cu only needs torch, c10 and
// the CUDA toolkit headers.
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <optional>
#include <string>

// Defined in fp8_gemm_kernels.cu (global namespace), signature unchanged.
void mma_emu_scaled_fp8_mm(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           std::optional<torch::Tensor> const& bias,
                           int64_t algorithm, int64_t f_bits, int64_t g_bits,
                           int64_t group_size, int64_t chunk_size);

// The upstream host wrapper launches with <<<grid, block>>> and does not check
// the launch status; expose cudaGetLastError so callers can detect a failed
// launch (e.g. missing kernel image) instead of reading an untouched output.
static std::string last_cuda_error() {
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return std::string();
  return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mma_emu_scaled_fp8_mm", &mma_emu_scaled_fp8_mm,
        "NADPE MMA-Emu scaled FP8 GEMM, writes c in place. "
        "a: fp8 e4m3 [M,K] row-major; b: fp8 e4m3 [K,N] column-major; "
        "scales: fp32 [1]; c: bf16/fp16 [M,N] row-major. "
        "algorithm 1=GDFS, 2=CoFDA C-fused, 3=CoFDA C-decoupled.",
        py::arg("c"), py::arg("a"), py::arg("b"), py::arg("a_scales"),
        py::arg("b_scales"), py::arg("bias"), py::arg("algorithm"),
        py::arg("f_bits"), py::arg("g_bits"), py::arg("group_size"),
        py::arg("chunk_size"));
  m.def("last_cuda_error", &last_cuda_error,
        "cudaGetLastError() as a string; empty when no error is pending.");
}
