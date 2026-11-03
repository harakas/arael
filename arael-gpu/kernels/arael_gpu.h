// arael_gpu.h: what differs between the targets of arael's GPU kernels.
//
// The runtime puts this in front of every kernel source, after the device's
// constants (ARAEL_WARP, ARAEL_SHARED_BYTES) and the kernel's own (SCALAR,
// tile sizes). Kernel code uses only the names defined here, the C++ subset
// of docs/dev/NATIVE_GPU_PLAN.md section 2, and the math functions every
// target has (sqrt, fabs, fma, isnan, min, max).
//
// Targets: CUDA (NVRTC). HIP and Metal add their branches in their stages.

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define ARAEL_CUDA 1
#else
#error "arael_gpu.h: no branch for this compiler"
#endif

typedef unsigned char u8;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;

#if ARAEL_CUDA

// An entry point keeps its name, so the runtime finds it in the module.
// KERNEL_THREADS(n): an entry point launched with at most n threads per block.
#define KERNEL extern "C" __global__
#define KERNEL_THREADS(n) extern "C" __global__ __launch_bounds__(n)
#define DEVICE __device__ inline

// A pointer parameter through which alone the kernel reaches what it points
// at; on a const pointer, the values are not written during the launch (the
// compiler may then read them through the read-only cache).
#define RESTRICT __restrict__

// Address spaces: empty on CUDA, Metal's qualifiers on Metal. SHARED
// declares an array in shared memory; SHARED_PTR qualifies a pointer to one.
#define GLOBAL
#define CONSTANT
#define SHARED __shared__
#define SHARED_PTR

// The launch's dynamic shared memory (Launch::shared_bytes) as an array of T.
#define DYNAMIC_SHARED(T, name) \
    extern __shared__ __align__(16) u8 arael_dynamic_shared[]; \
    SHARED_PTR T* name = (SHARED_PTR T*)arael_dynamic_shared

// BARRIER: every thread of the block. WARP_BARRIER: every lane of the warp.
#define BARRIER() __syncthreads()
#define WARP_BARRIER() __syncwarp()
#define HAS_F64 1

DEVICE u32 thread_x() { return threadIdx.x; }
DEVICE u32 block_x() { return blockIdx.x; }
DEVICE u32 block_dim_x() { return blockDim.x; }
DEVICE u32 global_x() { return blockIdx.x * blockDim.x + threadIdx.x; }

// `*p += v` as one indivisible step, in global or shared memory; returns the
// old value.
DEVICE float atomic_add(float* p, float v) { return atomicAdd(p, v); }
DEVICE double atomic_add(double* p, double v) { return atomicAdd(p, v); }
DEVICE u32 atomic_add(u32* p, u32 v) { return atomicAdd(p, v); }

// `v` of lane (this lane XOR mask). Every lane of the warp calls it.
DEVICE float shfl_xor(float v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE double shfl_xor(double v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }

// The sum of `v` over the warp, in every lane: a butterfly over lane
// distances 1, 2, 4, ... Every lane of the warp calls it.
template <typename T> DEVICE T warp_sum(T v) {
#pragma unroll
    for (u32 o = 1; o < ARAEL_WARP; o <<= 1) v += shfl_xor(v, o);
    return v;
}

// Rounded on their own: never fused with a neighbouring operation into an
// FMA. Double-single arithmetic (df32) needs these where a plain `+` or `*`
// may be contracted.
DEVICE float add_rn(float a, float b) { return __fadd_rn(a, b); }
DEVICE float sub_rn(float a, float b) { return __fsub_rn(a, b); }
DEVICE float mul_rn(float a, float b) { return __fmul_rn(a, b); }
DEVICE double add_rn(double a, double b) { return __dadd_rn(a, b); }
DEVICE double sub_rn(double a, double b) { return __dsub_rn(a, b); }
DEVICE double mul_rn(double a, double b) { return __dmul_rn(a, b); }

#endif
