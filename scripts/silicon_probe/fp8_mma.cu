// Native FP8 MMA witness. Fragment coordinates follow NVIDIA PTX ISA 9.7.16.5.10.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

__device__ __forceinline__ uint32_t pack4(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

__global__ void native_fp8_mma_kernel(const uint8_t* a, const uint8_t* bt,
                                      const float* c, float* d, int k) {
    const int tile = blockIdx.x;
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread = lane & 3;
    a += int64_t(tile) * 16 * k;
    bt += int64_t(tile) * 8 * k;
    c += int64_t(tile) * 16 * 8;
    d += int64_t(tile) * 16 * 8;
    // C/D: row = group + 8*(i/2), column = 2*thread + (i%2).
    float d0 = c[group * 8 + 2 * thread];
    float d1 = c[group * 8 + 2 * thread + 1];
    float d2 = c[(group + 8) * 8 + 2 * thread];
    float d3 = c[(group + 8) * 8 + 2 * thread + 1];
    for (int start = 0; start < k; start += 32) {
        // A packed words: rows group, group+8, group, group+8; K halves 0,0,16,16.
        const uint32_t a0 = pack4(a + group * k + start + 4 * thread);
        const uint32_t a1 = pack4(a + (group + 8) * k + start + 4 * thread);
        const uint32_t a2 = pack4(a + group * k + start + 4 * thread + 16);
        const uint32_t a3 = pack4(a + (group + 8) * k + start + 4 * thread + 16);
        // B is mathematically [K,8], supplied as contiguous B-transpose [8,K].
        const uint32_t b0 = pack4(bt + group * k + start + 4 * thread);
        const uint32_t b1 = pack4(bt + group * k + start + 4 * thread + 16);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    d[group * 8 + 2 * thread] = d0;
    d[group * 8 + 2 * thread + 1] = d1;
    d[(group + 8) * 8 + 2 * thread] = d2;
    d[(group + 8) * 8 + 2 * thread + 1] = d3;
}

torch::Tensor native_fp8_mma(torch::Tensor a, torch::Tensor bt, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && c.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(a.device() == bt.device() && a.device() == c.device(), "devices must agree");
    TORCH_CHECK(a.is_contiguous() && bt.is_contiguous() && c.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kUInt8 && bt.scalar_type() == torch::kUInt8,
                "multiplicands must contain raw E4M3 bytes");
    TORCH_CHECK(c.scalar_type() == torch::kFloat32, "C must be FP32");
    TORCH_CHECK(a.dim() == 3 && bt.dim() == 3 && c.dim() == 3, "inputs must be batched rank-3 tiles");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) == 16 && bt.size(1) == 8,
                "A/BT shapes must be [batch,16,K]/[batch,8,K]");
    TORCH_CHECK(a.size(0) == bt.size(0) && a.size(0) == c.size(0) &&
                a.size(2) == bt.size(2) && c.size(1) == 16 && c.size(2) == 8,
                "batch, K and C shapes must agree");
    TORCH_CHECK(a.size(2) > 0 && a.size(2) % 32 == 0 && a.size(2) <= 4096,
                "K must be a positive multiple of 32, no greater than 4096");
    TORCH_CHECK(a.size(0) <= 65535, "probe batch exceeds supported grid bound");
    c10::cuda::CUDAGuard guard(a.device());
    const auto* properties = at::cuda::getDeviceProperties(a.get_device());
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "this narrow probe is compiled for Hopper sm_90 only");
    auto d = torch::empty_like(c);
    native_fp8_mma_kernel<<<a.size(0), 32, 0, at::cuda::getCurrentCUDAStream(a.get_device())>>>(
        a.data_ptr<uint8_t>(), bt.data_ptr<uint8_t>(), c.data_ptr<float>(), d.data_ptr<float>(), int(a.size(2)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("native_fp8_mma", &native_fp8_mma, "Raw E4M3 native sm_90 MMA sequence");
}
