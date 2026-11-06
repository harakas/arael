// cpu.h: arael's GPU kernels on the CPU (arael_gpu::cpu), the kernels' side.
//
// The CPU device compiles a kernel source with the host's C++ compiler,
// this file and arael_gpu.h's CPU branch in front of it. The launches run
// in the runtime (cpu_runtime.cpp), a library of its own loaded once per
// process: a launch runs its blocks one after another on the calling
// thread, a block's threads as fibers that switch at barriers. A kernel
// library reaches the runtime through its table (Runtime), handed to
// arael_cpu_init when the library is loaded. Both are compiled with hidden
// visibility: a library exports its launchers alone and shares nothing
// with another.

#include <cmath>
#include <cstring>
#include <math.h>
#include <type_traits>
#include <utility>

#define ARAEL_CPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace arael_cpu {

// A lane's slot for a shuffle: the widest value shuffled.
constexpr unsigned LANE_BYTES = 16;

// The launch running on a thread, as a kernel sees it.
struct Run {
    unsigned grid[3];
    unsigned block[3];
    unsigned bid[3];
    unsigned threads;
    unsigned warp;
    // The current thread of the block.
    unsigned cur;
    unsigned char* lanes;
    unsigned char* dynamic_shared;
};

// What the runtime gives a kernel library.
struct Runtime {
    // The calling thread's launch.
    Run* (*current)();
    void (*block_barrier)();
    void (*warp_barrier)();
    // `body(arg)` as every thread of the launch `l` (grid, block, dynamic
    // shared bytes) with warps of `warp` lanes; 0 on success, else 1 with
    // `*error`.
    int (*launch)(void (*body)(void*), void* arg, const unsigned* l, unsigned warp, const char** error);
};

} // namespace arael_cpu

#ifndef ARAEL_CPU_RUNTIME

namespace arael_cpu {

static const Runtime* rt = nullptr;

inline void block_barrier() { rt->block_barrier(); }
inline void warp_barrier() { rt->warp_barrier(); }
inline unsigned thread_x() {
    Run* r = rt->current();
    return r->cur % r->block[0];
}
inline unsigned block_x() { return rt->current()->bid[0]; }
inline unsigned block_dim_x() { return rt->current()->block[0]; }
inline unsigned char* dynamic_shared() { return rt->current()->dynamic_shared; }
inline unsigned lane() { return rt->current()->cur % ARAEL_WARP; }

// `v` of lane `src` of this thread's warp, the lanes exchanging through
// their slots between two warp barriers.
template <class T> T exchange(T v, unsigned src) {
    static_assert(sizeof(T) <= LANE_BYTES, "a shuffled value fits a lane's slot");
    Run* r = rt->current();
    unsigned me = r->cur;
    memcpy(r->lanes + (size_t)me * LANE_BYTES, &v, sizeof(T));
    warp_barrier();
    unsigned base = me - me % ARAEL_WARP;
    T out;
    memcpy(&out, r->lanes + (size_t)(base + src % ARAEL_WARP) * LANE_BYTES, sizeof(T));
    warp_barrier();
    return out;
}

template <class... A, size_t... I>
void invoke(void (*k)(A...), void** p, std::index_sequence<I...>) {
    k(*static_cast<typename std::remove_reference<A>::type*>(p[I])...);
}

// Launch the kernel `k`: `p` its arguments, each in a slot of its own read
// as the parameter's type (the driver's convention), `l` the grid, the
// block and the dynamic shared bytes. 0 on success, else 1 with `*error`.
template <class... A>
int launch(void (*k)(A...), void** p, const unsigned* l, const char** error) {
    struct Call {
        void (*k)(A...);
        void** p;
    } call{k, p};
    auto body = [](void* c) {
        Call* call = static_cast<Call*>(c);
        invoke(call->k, call->p, std::index_sequence_for<A...>{});
    };
    return rt->launch(body, &call, l, ARAEL_WARP, error);
}

} // namespace arael_cpu

ARAEL_CPU_EXPORT void arael_cpu_init(const arael_cpu::Runtime* rt) { arael_cpu::rt = rt; }

#endif
