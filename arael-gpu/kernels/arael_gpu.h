// arael_gpu.h: what differs between the targets of arael's GPU kernels.
//
// The runtime puts this in front of every kernel source, after the device's
// constants (ARAEL_WARP, ARAEL_SHARED_BYTES) and the kernel's own (SCALAR,
// tile sizes). Kernel code uses only the names defined here, the C++ subset
// of docs/dev/NATIVE_GPU_PLAN.md section 2, and the math functions every
// target has (sqrt, fabs, fma, isnan, min, max).
//
// Targets: CUDA (NVRTC), HIP (hipRTC), Metal (Metal's run-time compiler,
// the Metal Shading Language) and the CPU device (the host's C++ compiler,
// cpu.h in front).

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define ARAEL_CUDA 1
#elif defined(__HIPCC_RTC__) || defined(__HIP__)
#define ARAEL_HIP 1
#elif defined(__METAL_VERSION__)
#define ARAEL_METAL 1
#elif defined(ARAEL_CPU_TARGET)
#define ARAEL_CPU 1
#else
#error "arael_gpu.h: no branch for this compiler"
#endif

#if ARAEL_METAL
#include <metal_stdlib>
using namespace metal;
// Metal has no `long long`; its `long` is 64 bits.
typedef uchar u8;
typedef uint u32;
typedef int i32;
typedef ulong u64;
typedef long i64;
#else
typedef unsigned char u8;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;
#endif

// A warp: the 32 lanes that shfl, shfl_xor, warp_sum and WARP_BARRIER span,
// threads 32 w to 32 w + 31 of a block. The device runs ARAEL_WARP lanes in
// lockstep: 32 on NVIDIA and AMD RDNA, 64 on AMD GCN and CDNA, whose
// wavefront holds two warps.
#define WARP 32u
#if ARAEL_WARP % 32 != 0
#error "arael_gpu.h: the device's lockstep width is not a whole number of 32-lane warps"
#endif

// UNROLL_32(i, body): `body` 32 times, each a block of its own in which `i`
// is a constant, 0 to 31. For a loop indexing a per-thread array, which
// stays in registers only when every index into it is a constant: Metal's
// compiler leaves big `#pragma unroll` nests rolled, and the array then
// lives in memory. The body may hold another UNROLL_32, not a preprocessor
// directive.
//
//     UNROLL_32(k, v[k] *= d[k]; UNROLL_32(c, if (c > k) v[c] -= v[k] * l[k * 32 + c];))
#define ARAEL_UNROLL_4(i, b, ...)                                                                                          \
    { const u32 i = (b) + 0u; __VA_ARGS__ } { const u32 i = (b) + 1u; __VA_ARGS__ } { const u32 i = (b) + 2u; __VA_ARGS__ } \
    { const u32 i = (b) + 3u; __VA_ARGS__ }
#define UNROLL_32(i, ...)                                                                                                \
    ARAEL_UNROLL_4(i, 0u, __VA_ARGS__) ARAEL_UNROLL_4(i, 4u, __VA_ARGS__) ARAEL_UNROLL_4(i, 8u, __VA_ARGS__)               \
    ARAEL_UNROLL_4(i, 12u, __VA_ARGS__) ARAEL_UNROLL_4(i, 16u, __VA_ARGS__) ARAEL_UNROLL_4(i, 20u, __VA_ARGS__)            \
    ARAEL_UNROLL_4(i, 24u, __VA_ARGS__) ARAEL_UNROLL_4(i, 28u, __VA_ARGS__)

#if ARAEL_CUDA || ARAEL_HIP

// An entry point keeps its name, so the runtime finds it in the module.
// KERNEL_THREADS(n): an entry point launched with at most n threads per block.
// KERNEL_THREADS_REGS(n): the same, its registers not capped to fit several
// blocks on a multiprocessor (one block must fit), for a kernel whose
// per-thread arrays would otherwise spill to local memory.
// KERNEL_THREADS_BLOCKS(n, b): the same, its registers capped so that b
// blocks fit a multiprocessor, for a kernel whose occupancy measures worth
// more than a spill (CUDA; on HIP as KERNEL_THREADS).
// (HIP's second launch bound is a minimum of waves per unit, and its
// default leaves the registers uncapped.)
#define KERNEL extern "C" __global__
#define KERNEL_THREADS(n) extern "C" __global__ __launch_bounds__(n)
#if ARAEL_CUDA
#define KERNEL_THREADS_REGS(n) extern "C" __global__ __launch_bounds__(n, 1)
#define KERNEL_THREADS_BLOCKS(n, b) extern "C" __global__ __launch_bounds__(n, b)
#else
#define KERNEL_THREADS_REGS(n) extern "C" __global__ __launch_bounds__(n)
#define KERNEL_THREADS_BLOCKS(n, b) extern "C" __global__ __launch_bounds__(n)
#endif
#define DEVICE __device__ inline
// DEVICE_INLINE: a device function always inlined, for one whose array
// parameters must stay in registers (a call keeps them in memory) and
// which has more than one caller, where the compiler may not inline it.
#if ARAEL_CUDA
#define DEVICE_INLINE __device__ __forceinline__
#else
#define DEVICE_INLINE __device__ inline __attribute__((always_inline))
#endif

// A pointer parameter through which alone the kernel reaches what it points
// at; on a const pointer, the values are not written during the launch (the
// compiler may then read them through the read-only cache).
#define RESTRICT __restrict__

// Address spaces: empty on CUDA and HIP, Metal's qualifiers on Metal.
// SHARED declares an array in shared memory; SHARED_PTR qualifies a pointer
// to one. THREAD qualifies a reference or pointer to the thread's own
// variables, which Metal requires of every reference parameter.
#define GLOBAL
#define CONSTANT
#define SHARED __shared__
#define SHARED_PTR
#define THREAD

// The launch's dynamic shared memory (Launch::shared_bytes) as an array of T.
#define DYNAMIC_SHARED(T, name) \
    extern __shared__ __attribute__((aligned(16))) u8 arael_dynamic_shared[]; \
    SHARED_PTR T* name = (SHARED_PTR T*)arael_dynamic_shared

// BARRIER: every thread of the block. WARP_BARRIER: every lane of the warp,
// their shared-memory writes seen by each other (on AMD, whose wavefront
// runs in lockstep, a wavefront-scope fence around the wave barrier, as
// HIP's __syncwarp).
#define BARRIER() __syncthreads()
#if ARAEL_CUDA
#define WARP_BARRIER() __syncwarp()
#else
#define WARP_BARRIER()                                    \
    do {                                                  \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); \
        __builtin_amdgcn_wave_barrier();                  \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); \
    } while (0)
#endif
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

// `v` of lane (this lane XOR mask), `mask` below 32, and `v` of lane `src`
// mod 32, within the warp. Every lane of the warp calls them. (On HIP the
// width WARP keeps each warp of a 64-lane wavefront to its own lanes.)
#if ARAEL_CUDA
DEVICE float shfl_xor(float v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE double shfl_xor(double v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE u32 shfl_xor(u32 v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE float shfl(float v, u32 src) { return __shfl_sync(0xffffffffu, v, (int)src); }
DEVICE double shfl(double v, u32 src) { return __shfl_sync(0xffffffffu, v, (int)src); }
#else
DEVICE float shfl_xor(float v, u32 mask) { return __shfl_xor(v, (int)mask, (int)WARP); }
DEVICE double shfl_xor(double v, u32 mask) { return __shfl_xor(v, (int)mask, (int)WARP); }
DEVICE u32 shfl_xor(u32 v, u32 mask) { return __shfl_xor(v, (int)mask, (int)WARP); }
DEVICE float shfl(float v, u32 src) { return __shfl(v, (int)src, (int)WARP); }
DEVICE double shfl(double v, u32 src) { return __shfl(v, (int)src, (int)WARP); }
#endif

// warp_uniform(v): v, the same in every lane of the device's lockstep unit
// (a CUDA warp, an AMD wavefront, a Metal SIMD group), as its first lane
// holds it, so the compiler treats it as uniform: branches on what is
// derived from it stay branches, where a value it cannot prove uniform (from
// thread_x(), say) turns them into predicated instructions that all issue.
// (Not `uniform`, a type of Metal's.)
#if ARAEL_CUDA
DEVICE u32 warp_uniform(u32 v) { return __shfl_sync(0xffffffffu, v, 0); }
#else
DEVICE u32 warp_uniform(u32 v) { return (u32)__builtin_amdgcn_readfirstlane((int)v); }
#endif

// The sum of `v` over the warp, in every lane: a butterfly over lane
// distances 1, 2, 4, ... Every lane of the warp calls it.
template <typename T> DEVICE T warp_sum(T v) {
#pragma unroll
    for (u32 o = 1; o < WARP; o <<= 1) v += shfl_xor(v, o);
    return v;
}

// Rounded on their own: never fused with a neighbouring operation into an
// FMA. Double-single arithmetic (df32) needs these where a plain `+` or `*`
// may be contracted. (HIP's __fadd_rn and kin are a plain `+` there, which
// its default contraction may fuse; the pragma keeps these apart.)
#if ARAEL_CUDA
DEVICE float add_rn(float a, float b) { return __fadd_rn(a, b); }
DEVICE float sub_rn(float a, float b) { return __fsub_rn(a, b); }
DEVICE float mul_rn(float a, float b) { return __fmul_rn(a, b); }
DEVICE double add_rn(double a, double b) { return __dadd_rn(a, b); }
DEVICE double sub_rn(double a, double b) { return __dsub_rn(a, b); }
DEVICE double mul_rn(double a, double b) { return __dmul_rn(a, b); }
#else
#define ARAEL_UNFUSED(T, name, op)          \
    DEVICE T name(T a, T b) {               \
        _Pragma("clang fp contract(off)")   \
        return a op b;                      \
    }
ARAEL_UNFUSED(float, add_rn, +)
ARAEL_UNFUSED(float, sub_rn, -)
ARAEL_UNFUSED(float, mul_rn, *)
ARAEL_UNFUSED(double, add_rn, +)
ARAEL_UNFUSED(double, sub_rn, -)
ARAEL_UNFUSED(double, mul_rn, *)
#undef ARAEL_UNFUSED
#endif

#endif

#if ARAEL_METAL

// The same names in the Metal Shading Language. The runtime rewrites each
// entry's parameters into Metal's buffer bindings and adds the parameters
// the index macros read (arael_tid, arael_bid, arael_bdim) and the dynamic
// shared memory (arael_dyn): src/metal.rs. So thread_x() and its kin work
// in kernel entries only, and a helper takes the index it needs as a
// parameter. Metal has no f64. A SIMD group is a warp, 32 lanes on every
// Apple GPU. There is no occupancy hint beyond the threads per block.
#define KERNEL kernel
#define KERNEL_THREADS(n) [[max_total_threads_per_threadgroup(n)]] kernel
#define KERNEL_THREADS_REGS(n) KERNEL_THREADS(n)
#define KERNEL_THREADS_BLOCKS(n, b) KERNEL_THREADS(n)
#define DEVICE inline
#define DEVICE_INLINE inline __attribute__((always_inline))
#define RESTRICT __restrict
#define GLOBAL device
#define CONSTANT constant
#define SHARED threadgroup
#define SHARED_PTR threadgroup
#define THREAD thread
#define DYNAMIC_SHARED(T, name) threadgroup T* name = (threadgroup T*)arael_dyn
// Device memory in both: kernels read what other threads of the block wrote
// to global memory before the barrier, as CUDA's barriers allow.
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define WARP_BARRIER() simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define HAS_F64 0

#define thread_x() (arael_tid.x)
#define block_x() (arael_bid.x)
#define block_dim_x() (arael_bdim.x)
#define global_x() (arael_bid.x * arael_bdim.x + arael_tid.x)

DEVICE float atomic_add(device float* p, float v) { return atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed); }
DEVICE u32 atomic_add(device u32* p, u32 v) { return atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed); }

DEVICE float shfl_xor(float v, u32 mask) { return simd_shuffle_xor(v, (ushort)mask); }
DEVICE u32 shfl_xor(u32 v, u32 mask) { return simd_shuffle_xor(v, (ushort)mask); }
DEVICE float shfl(float v, u32 src) { return simd_shuffle(v, (ushort)(src % WARP)); }
DEVICE u32 warp_uniform(u32 v) { return simd_broadcast_first(v); }

template <typename T> DEVICE T warp_sum(T v) {
#pragma unroll
    for (u32 o = 1; o < WARP; o <<= 1) v += shfl_xor(v, o);
    return v;
}

// Metal contracts `a * b + c` by default; under the pragma a product or sum
// is rounded on its own.
#define ARAEL_UNFUSED(T, name, op)          \
    DEVICE T name(T a, T b) {               \
        _Pragma("clang fp contract(off)")   \
        return a op b;                      \
    }
ARAEL_UNFUSED(float, add_rn, +)
ARAEL_UNFUSED(float, sub_rn, -)
ARAEL_UNFUSED(float, mul_rn, *)
#undef ARAEL_UNFUSED

DEVICE float __int_as_float(int v) { return as_type<float>(v); }

// What CUDA's library has and Metal's lacks: log(1 + x) and exp(x) - 1
// without the cancellation of 1 + x and exp(x) - 1 near 0 (Kahan's forms:
// the rounding of u = 1 + x, or of exp(x), divided out).
DEVICE float log1p(float x) {
    float u = 1.0f + x;
    if (u == 1.0f || isinf(u)) return u == 1.0f ? x : u;
    return log(u) * (x / (u - 1.0f));
}
DEVICE float expm1(float x) {
    float u = exp(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f || isinf(u)) return um1;
    return um1 * x / log(u);
}

#endif

#if ARAEL_CPU

// The same names over cpu.h. A block runs on one thread, its threads as
// fibers that switch only at barriers, so an atomic is a plain read and
// write. Shared memory is a plain static of the kernel's library: each
// device loads a library of its own and runs its launches one at a time
// (its stream lock), so no two blocks use it at once. Not `thread_local`:
// macOS gives every loaded library with thread-locals one of the
// process's 512 pthread keys, and dyld aborts past them, while a process
// loads a library per device and source. The unit is compiled with
// contraction on and FMA, as the GPU compilers' default.
#define KERNEL extern "C"
#define KERNEL_THREADS(n) extern "C"
#define KERNEL_THREADS_REGS(n) extern "C"
#define KERNEL_THREADS_BLOCKS(n, b) extern "C"
#define DEVICE inline
#define DEVICE_INLINE inline
#define RESTRICT __restrict__
#define GLOBAL
#define CONSTANT
#define SHARED static
#define SHARED_PTR
#define THREAD
#define DYNAMIC_SHARED(T, name) T* name = (T*)arael_cpu::dynamic_shared()
#define BARRIER() arael_cpu::block_barrier()
#define WARP_BARRIER() arael_cpu::warp_barrier()
#define HAS_F64 1

DEVICE u32 thread_x() { return arael_cpu::thread_x(); }
DEVICE u32 block_x() { return arael_cpu::block_x(); }
DEVICE u32 block_dim_x() { return arael_cpu::block_dim_x(); }
DEVICE u32 global_x() { return arael_cpu::block_x() * arael_cpu::block_dim_x() + arael_cpu::thread_x(); }

// `v` takes the pointee's type, as CUDA's overloads convert it.
template <typename T> struct arael_same {
    typedef T type;
};
template <typename T> DEVICE T atomic_add(T* p, typename arael_same<T>::type v) {
    T old = *p;
    *p = old + v;
    return old;
}

DEVICE float shfl_xor(float v, u32 mask) { return arael_cpu::exchange(v, arael_cpu::lane() ^ mask); }
DEVICE double shfl_xor(double v, u32 mask) { return arael_cpu::exchange(v, arael_cpu::lane() ^ mask); }
DEVICE u32 shfl_xor(u32 v, u32 mask) { return arael_cpu::exchange(v, arael_cpu::lane() ^ mask); }
DEVICE float shfl(float v, u32 src) { return arael_cpu::exchange(v, src); }
DEVICE double shfl(double v, u32 src) { return arael_cpu::exchange(v, src); }
DEVICE u32 warp_uniform(u32 v) { return v; }

template <typename T> DEVICE T warp_sum(T v) {
    for (u32 o = 1; o < WARP; o <<= 1) v += shfl_xor(v, o);
    return v;
}

// The operands and the result pass through an empty asm, which the
// compiler cannot see through, so the operation is not fused with another.
#if defined(__x86_64__)
#define ARAEL_PIN(v) __asm__("" : "+x"(v))
#elif defined(__aarch64__)
#define ARAEL_PIN(v) __asm__("" : "+w"(v))
#else
#error "arael_gpu.h: no ARAEL_PIN for this CPU"
#endif
#define ARAEL_UNFUSED(T, name, op) \
    DEVICE T name(T a, T b) {      \
        ARAEL_PIN(a);              \
        ARAEL_PIN(b);              \
        T r = a op b;              \
        ARAEL_PIN(r);              \
        return r;                  \
    }
ARAEL_UNFUSED(float, add_rn, +)
ARAEL_UNFUSED(float, sub_rn, -)
ARAEL_UNFUSED(float, mul_rn, *)
ARAEL_UNFUSED(double, add_rn, +)
ARAEL_UNFUSED(double, sub_rn, -)
ARAEL_UNFUSED(double, mul_rn, *)
#undef ARAEL_UNFUSED
#undef ARAEL_PIN

// What CUDA and HIP give kernels beyond the C++ library: min and max over
// mixed types as CUDA resolves them (the common type; the floating ones
// through fmin and fmax, a NaN operand losing), rsqrt and the bit casts.
template <typename A, typename B> DEVICE typename std::common_type<A, B>::type min(A a, B b) {
    typedef typename std::common_type<A, B>::type C;
    if constexpr (std::is_floating_point<C>::value) return std::fmin((C)a, (C)b);
    else return (C)a < (C)b ? (C)a : (C)b;
}
template <typename A, typename B> DEVICE typename std::common_type<A, B>::type max(A a, B b) {
    typedef typename std::common_type<A, B>::type C;
    if constexpr (std::is_floating_point<C>::value) return std::fmax((C)a, (C)b);
    else return (C)a < (C)b ? (C)b : (C)a;
}
DEVICE float rsqrt(float x) { return 1.0f / std::sqrt(x); }
DEVICE double rsqrt(double x) { return 1.0 / std::sqrt(x); }
DEVICE float __int_as_float(int v) {
    float f;
    memcpy(&f, &v, sizeof f);
    return f;
}
DEVICE double __longlong_as_double(long long v) {
    double d;
    memcpy(&d, &v, sizeof d);
    return d;
}
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::copysign;
using std::cos;
using std::cosh;
using std::exp;
using std::expm1;
using std::fabs;
using std::floor;
using std::fma;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::log;
using std::log10;
using std::log1p;
using std::log2;
using std::pow;
using std::round;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

#endif
