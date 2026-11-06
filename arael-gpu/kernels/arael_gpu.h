// arael_gpu.h: what differs between the targets of arael's GPU kernels.
//
// The runtime puts this in front of every kernel source, after the device's
// constants (ARAEL_WARP, ARAEL_SHARED_BYTES) and the kernel's own (SCALAR,
// tile sizes). Kernel code uses only the names defined here, the C++ subset
// of docs/dev/NATIVE_GPU_PLAN.md section 2, and the math functions every
// target has (sqrt, fabs, fma, isnan, min, max).
//
// Targets: CUDA (NVRTC), HIP (hipRTC) and the CPU device (the host's C++
// compiler, cpu.h in front). Metal adds its branch in its stage.

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define ARAEL_CUDA 1
#elif defined(__HIPCC_RTC__) || defined(__HIP__)
#define ARAEL_HIP 1
#elif defined(ARAEL_CPU_TARGET)
#define ARAEL_CPU 1
#else
#error "arael_gpu.h: no branch for this compiler"
#endif

typedef unsigned char u8;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;

#if ARAEL_CUDA || ARAEL_HIP

// An entry point keeps its name, so the runtime finds it in the module.
// KERNEL_THREADS(n): an entry point launched with at most n threads per block.
// KERNEL_THREADS_REGS(n): the same, its registers not capped to fit several
// blocks on a multiprocessor (one block must fit), for a kernel whose
// per-thread arrays would otherwise spill to local memory. (HIP's second
// launch bound is a minimum of waves per unit, and its default leaves the
// registers uncapped.)
#define KERNEL extern "C" __global__
#define KERNEL_THREADS(n) extern "C" __global__ __launch_bounds__(n)
#if ARAEL_CUDA
#define KERNEL_THREADS_REGS(n) extern "C" __global__ __launch_bounds__(n, 1)
#else
#define KERNEL_THREADS_REGS(n) extern "C" __global__ __launch_bounds__(n)
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
// to one.
#define GLOBAL
#define CONSTANT
#define SHARED __shared__
#define SHARED_PTR

// The launch's dynamic shared memory (Launch::shared_bytes) as an array of T.
#define DYNAMIC_SHARED(T, name) \
    extern __shared__ __attribute__((aligned(16))) u8 arael_dynamic_shared[]; \
    SHARED_PTR T* name = (SHARED_PTR T*)arael_dynamic_shared

// BARRIER: every thread of the block. WARP_BARRIER: every lane of the warp,
// their shared-memory writes seen by each other (on AMD a wavefront-scope
// fence around the wave barrier, as HIP's __syncwarp).
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

// `v` of lane (this lane XOR mask), and `v` of lane `src`. Every lane of
// the warp calls them.
#if ARAEL_CUDA
DEVICE float shfl_xor(float v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE double shfl_xor(double v, u32 mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
DEVICE float shfl(float v, u32 src) { return __shfl_sync(0xffffffffu, v, (int)src); }
DEVICE double shfl(double v, u32 src) { return __shfl_sync(0xffffffffu, v, (int)src); }
#else
DEVICE float shfl_xor(float v, u32 mask) { return __shfl_xor(v, (int)mask); }
DEVICE double shfl_xor(double v, u32 mask) { return __shfl_xor(v, (int)mask); }
DEVICE float shfl(float v, u32 src) { return __shfl(v, (int)src); }
DEVICE double shfl(double v, u32 src) { return __shfl(v, (int)src); }
#endif

// The sum of `v` over the warp, in every lane: a butterfly over lane
// distances 1, 2, 4, ... Every lane of the warp calls it.
template <typename T> DEVICE T warp_sum(T v) {
#pragma unroll
    for (u32 o = 1; o < ARAEL_WARP; o <<= 1) v += shfl_xor(v, o);
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

#if ARAEL_CPU

// The same names over cpu.h. A block runs on one thread, its threads as
// fibers that switch only at barriers, so shared memory is a static of
// that thread and an atomic is a plain read and write. The unit is
// compiled with contraction on and FMA, as the GPU compilers' default.
#define KERNEL extern "C"
#define KERNEL_THREADS(n) extern "C"
#define KERNEL_THREADS_REGS(n) extern "C"
#define DEVICE inline
#define DEVICE_INLINE inline
#define RESTRICT __restrict__
#define GLOBAL
#define CONSTANT
#define SHARED static thread_local
#define SHARED_PTR
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
DEVICE float shfl(float v, u32 src) { return arael_cpu::exchange(v, src); }
DEVICE double shfl(double v, u32 src) { return arael_cpu::exchange(v, src); }

template <typename T> DEVICE T warp_sum(T v) {
    for (u32 o = 1; o < ARAEL_WARP; o <<= 1) v += shfl_xor(v, o);
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
