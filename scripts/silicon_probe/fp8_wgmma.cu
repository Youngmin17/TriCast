// Independent raw-FP8 WGMMA probe; NVIDIA PTX ISA 9.7.17.5 and 9.7.17.7.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

// K-major, no swizzle: ((8,m),(16,2)):((16,256),(1,128)).
// Offsets are bytes because A/B consist of unmodified eight-bit E4M3 codes.
__device__ __forceinline__ int fp8_smem_offset(int row, int k) {
    return (row % 8) * 16 + (row / 8) * 256 + (k % 16) + (k / 16) * 128;
}

__device__ __forceinline__ uint64_t fp8_descriptor(const uint8_t* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    // PTX: encode(x)=(x & 0x3ffff)>>4; LBO=128, SBO=256, swizzle=0.
    return uint64_t((address & 0x3ffff) >> 4) |
           (uint64_t(128 >> 4) << 16) | (uint64_t(256 >> 4) << 32);
}

__global__ void native_fp8_wgmma_kernel(const uint8_t* a, const uint8_t* bt,
                                       const float* c, float* d, int k) {
    __shared__ __align__(128) uint8_t shared_a[64 * 32];
    __shared__ __align__(128) uint8_t shared_b[8 * 32];
    const int tid = threadIdx.x;  // Exactly one complete 128-thread warpgroup.
    const int lane = tid % 32;
    const int row = 16 * (tid / 32) + lane / 4;
    const int column = 2 * (lane % 4);
    a += int64_t(blockIdx.x) * 64 * k;
    bt += int64_t(blockIdx.x) * 8 * k;
    c += int64_t(blockIdx.x) * 64 * 8;
    d += int64_t(blockIdx.x) * 64 * 8;
    float d0 = c[row * 8 + column];
    float d1 = c[row * 8 + column + 1];
    float d2 = c[(row + 8) * 8 + column];
    float d3 = c[(row + 8) * 8 + column + 1];
    const uint64_t desc_a = fp8_descriptor(shared_a);
    const uint64_t desc_b = fp8_descriptor(shared_b);
    // All control flow and barriers are uniform across the complete warpgroup.
    for (int begin = 0; begin < k; begin += 32) {
        for (int i = tid; i < 64 * 32; i += 128) {
            const int r = i / 32, kk = i % 32;
            shared_a[fp8_smem_offset(r, kk)] = a[r * k + begin + kk];
        }
        for (int i = tid; i < 8 * 32; i += 128) {
            const int r = i / 32, kk = i % 32;
            shared_b[fp8_smem_offset(r, kk)] = bt[r * k + begin + kk];
        }
        // Ordinary shared stores must become visible to the async WGMMA proxy.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        asm volatile(
            "{\n"
            ".reg .pred accumulate;\n"
            "setp.ne.b32 accumulate, 1, 0;\n"
            "wgmma.fence.sync.aligned;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3}, %4, %5, accumulate, 1, 1;\n"
            "wgmma.commit_group.sync.aligned;\n"
            "wgmma.wait_group.sync.aligned 0;\n"
            "}\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "l"(desc_a), "l"(desc_b) : "memory");
        // No producer may overwrite A/B before all warps finish this chunk.
        __syncthreads();
    }
    d[row * 8 + column] = d0;
    d[row * 8 + column + 1] = d1;
    d[(row + 8) * 8 + column] = d2;
    d[(row + 8) * 8 + column + 1] = d3;
}

torch::Tensor native_fp8_wgmma(torch::Tensor a, torch::Tensor bt, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && bt.is_cuda() && c.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(a.device() == bt.device() && a.device() == c.device(), "devices must agree");
    TORCH_CHECK(a.is_contiguous() && bt.is_contiguous() && c.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kUInt8 && bt.scalar_type() == torch::kUInt8,
                "multiplicands must contain raw E4M3 bytes");
    TORCH_CHECK(c.scalar_type() == torch::kFloat32, "C must be FP32");
    TORCH_CHECK(a.dim() == 3 && bt.dim() == 3 && c.dim() == 3, "inputs must be batched rank-3 tiles");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) == 64 && bt.size(1) == 8,
                "A/BT shapes must be [batch,64,K]/[batch,8,K]");
    TORCH_CHECK(a.size(0) == bt.size(0) && a.size(0) == c.size(0) &&
                a.size(2) == bt.size(2) && c.size(1) == 64 && c.size(2) == 8,
                "batch, K and C shapes must agree");
    TORCH_CHECK(a.size(2) > 0 && a.size(2) % 32 == 0 && a.size(2) <= 4096,
                "K must be a positive multiple of 32, no greater than 4096");
    TORCH_CHECK(a.size(0) <= 65535, "probe batch exceeds supported grid bound");
    c10::cuda::CUDAGuard guard(a.device());
    const auto* properties = at::cuda::getDeviceProperties(a.get_device());
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "this narrow probe requires Hopper and is compiled only for sm_90a");
    auto d = torch::empty_like(c);
    native_fp8_wgmma_kernel<<<a.size(0), 128, 0, at::cuda::getCurrentCUDAStream(a.get_device())>>>(
        a.data_ptr<uint8_t>(), bt.data_ptr<uint8_t>(), c.data_ptr<float>(), d.data_ptr<float>(), int(a.size(2)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("native_fp8_wgmma", &native_fp8_wgmma, "Raw E4M3 sm_90a WGMMA sequence with FP32 C/D");
}
