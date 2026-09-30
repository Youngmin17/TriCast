// Standalone pybind11 binding for the NADPE MMA-Emu NVFP4 / MXFP4 operators.
//
// Compiled together with the UNMODIFIED upstream sources
//   micro26-ae/csrc/quantization/mma_emu/{nvfp4,mxfp4}_gemm_kernels.cu
// (Apache-2.0). Upstream only builds these for SM100+ with CUDA >= 12.8; this
// binding exists to test whether the emulation itself also builds for sm_80.
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <string>

void mma_emu_scaled_nvfp4_mm(torch::Tensor& D, torch::Tensor const& A,
                             torch::Tensor const& B, torch::Tensor const& A_sf,
                             torch::Tensor const& B_sf,
                             torch::Tensor const& alpha, int64_t algorithm,
                             int64_t f_bits, int64_t g_bits);

void mma_emu_scaled_mxfp4_mm(torch::Tensor& D, torch::Tensor const& A,
                             torch::Tensor const& B, torch::Tensor const& A_sf,
                             torch::Tensor const& B_sf, int64_t algorithm,
                             int64_t f_bits, int64_t g_bits,
                             int64_t group_size, int64_t chunk_size);

static std::string last_cuda_error() {
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) return std::string();
  return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mma_emu_scaled_nvfp4_mm", &mma_emu_scaled_nvfp4_mm,
        "NADPE MMA-Emu NVFP4 GEMM (writes D in place)", py::arg("D"),
        py::arg("A"), py::arg("B"), py::arg("A_sf"), py::arg("B_sf"),
        py::arg("alpha"), py::arg("algorithm"), py::arg("f_bits"),
        py::arg("g_bits"));
  m.def("mma_emu_scaled_mxfp4_mm", &mma_emu_scaled_mxfp4_mm,
        "NADPE MMA-Emu MXFP4 GEMM (writes D in place)", py::arg("D"),
        py::arg("A"), py::arg("B"), py::arg("A_sf"), py::arg("B_sf"),
        py::arg("algorithm"), py::arg("f_bits"), py::arg("g_bits"),
        py::arg("group_size"), py::arg("chunk_size"));
  m.def("last_cuda_error", &last_cuda_error,
        "cudaGetLastError() as a string; empty when no error is pending.");
}
