// cpu_runtime.cpp: the runtime of arael's GPU kernels on the CPU
// (arael_gpu::cpu), compiled with cpu.h in front into a library loaded
// once per process and never unloaded.
//
// A launch runs its blocks one after another on the calling thread; a
// block's threads are fibers, each on a stack of its own. A fiber runs
// until it reaches a barrier, then the next one runs: BARRIER waits for
// every live thread of the block, WARP_BARRIER for every live lane of its
// warp, as on the GPU (a thread that has returned no longer takes part). A
// barrier that some live thread never reaches ends the launch with an
// error.
//
// A fiber starts once (makecontext) and then runs one GPU thread after
// another, waiting between them in this library's code, switching with
// _setjmp and _longjmp (no system call). The library is compiled without
// _FORTIFY_SOURCE, whose longjmp check refuses a jump to another stack.

#include <csetjmp>
#include <cstdlib>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

namespace arael_cpu {

// A fiber's stack, below it a guard page that faults on an overflow. The
// guard is the system's page (16 KB on Apple silicon): mprotect works in
// whole pages, so a smaller guard would protect the stack's first bytes.
constexpr size_t STACK_BYTES = 256 * 1024;
const size_t GUARD_BYTES = (size_t)sysconf(_SC_PAGESIZE);

enum State : unsigned char { READY, AT_BLOCK, AT_WARP, DONE };

struct Fiber {
    // Where it goes on: at a barrier, or done and waiting for its next thread.
    jmp_buf jb;
    ucontext_t start;
    State state;
    // Started: waiting in `jb`; else it starts at fiber_main.
    bool started;
    // The barrier generation it waits to see pass.
    unsigned gen;
};

// A launch: what kernels see, and the scheduler's state.
struct Launch {
    Run run;
    void (*body)(void*);
    void* arg;
    jmp_buf sched;
    // The block barrier: live threads, arrived, generation.
    unsigned live, arrived, gen;
    // Per warp: live lanes, arrived, generation.
    unsigned* wlive;
    unsigned* warrived;
    unsigned* wgen;
    Fiber* fibers;
    char** stacks;
    const char* error;
};

// What a thread keeps between launches: the fibers, their stacks and the
// launch's tables, grown to the largest launch, released when the thread ends.
struct Store {
    Fiber* fibers = nullptr;
    char** stacks = nullptr;
    unsigned nfibers = 0;
    unsigned* warps = nullptr;
    unsigned char* lanes = nullptr;
    unsigned cap = 0;
    unsigned char* shared = nullptr;
    unsigned shared_cap = 0;

    ~Store() {
        for (unsigned i = 0; i < nfibers; i++) munmap(stacks[i], GUARD_BYTES + STACK_BYTES);
        free(fibers);
        free(stacks);
        free(warps);
        free(lanes);
        free(shared);
    }
};

thread_local Launch* now = nullptr;
thread_local Store store;

void yield(Launch* l, Fiber& f) {
    if (!_setjmp(f.jb)) _longjmp(l->sched, 1);
}

// The current thread has returned: it leaves the barriers, and its fiber
// waits for its next thread.
void finish(Launch* l) {
    unsigned cur = l->run.cur;
    Fiber& f = l->fibers[cur];
    unsigned w = cur / l->run.warp;
    l->live--;
    l->wlive[w]--;
    // the others at a barrier no longer wait for this thread
    if (l->arrived > 0 && l->arrived == l->live) {
        l->arrived = 0;
        l->gen++;
    }
    if (l->warrived[w] > 0 && l->warrived[w] == l->wlive[w]) {
        l->warrived[w] = 0;
        l->wgen[w]++;
    }
    f.state = DONE;
    yield(l, f);
}

void fiber_main() {
    for (;;) {
        Launch* l = now;
        l->body(l->arg);
        finish(l);
    }
}

Run* current() { return &now->run; }

void block_barrier() {
    Launch* l = now;
    Fiber& f = l->fibers[l->run.cur];
    if (++l->arrived == l->live) {
        l->arrived = 0;
        l->gen++;
        return;
    }
    f.state = AT_BLOCK;
    f.gen = l->gen;
    yield(l, f);
}

void warp_barrier() {
    Launch* l = now;
    Fiber& f = l->fibers[l->run.cur];
    unsigned w = l->run.cur / l->run.warp;
    if (++l->warrived[w] == l->wlive[w]) {
        l->warrived[w] = 0;
        l->wgen[w]++;
        return;
    }
    f.state = AT_WARP;
    f.gen = l->wgen[w];
    yield(l, f);
}

// One block: its fibers run in turns until every thread has returned;
// false when none can go on (a barrier some live thread never reaches).
bool run_block(Launch* l) {
    unsigned n = l->run.threads, warp = l->run.warp;
    unsigned warps = (n + warp - 1) / warp;
    l->live = n;
    l->arrived = 0;
    l->gen = 0;
    for (unsigned w = 0; w < warps; w++) {
        unsigned first = w * warp;
        l->wlive[w] = n - first < warp ? n - first : warp;
        l->warrived[w] = 0;
        l->wgen[w] = 0;
    }
    for (unsigned i = 0; i < n; i++) l->fibers[i].state = READY;
    for (;;) {
        bool left = false, moved = false;
        for (unsigned i = 0; i < n; i++) {
            Fiber& f = l->fibers[i];
            if (f.state == DONE) continue;
            left = true;
            if (f.state == AT_BLOCK && f.gen == l->gen) continue;
            if (f.state == AT_WARP && f.gen == l->wgen[i / warp]) continue;
            f.state = READY;
            l->run.cur = i;
            moved = true;
            if (!_setjmp(l->sched)) {
                if (!f.started) {
                    f.started = true;
                    getcontext(&f.start);
                    f.start.uc_stack.ss_sp = l->stacks[i] + GUARD_BYTES;
                    f.start.uc_stack.ss_size = STACK_BYTES;
                    f.start.uc_link = nullptr;
                    makecontext(&f.start, fiber_main, 0);
                    setcontext(&f.start);
                }
                _longjmp(f.jb, 1);
            }
        }
        if (!left) return true;
        if (!moved) {
            // the stuck fibers start afresh at the next launch
            for (unsigned i = 0; i < n; i++)
                if (l->fibers[i].state != DONE) l->fibers[i].started = false;
            l->error = "a barrier that some live thread of the block never reaches";
            return false;
        }
    }
}

// The store grown to `threads` fibers and `shared` bytes of dynamic shared
// memory; false when memory runs out.
bool reserve(unsigned threads, unsigned shared) {
    Store& s = store;
    if (threads > s.cap) {
        unsigned* warps = (unsigned*)realloc(s.warps, sizeof(unsigned) * 3 * threads);
        if (warps) s.warps = warps;
        unsigned char* lanes = (unsigned char*)realloc(s.lanes, (size_t)LANE_BYTES * threads);
        if (lanes) s.lanes = lanes;
        // a started fiber's jmp_buf moves with it: its stack stays put
        Fiber* fibers = (Fiber*)realloc((void*)s.fibers, sizeof(Fiber) * threads);
        if (fibers) s.fibers = fibers;
        char** stacks = (char**)realloc(s.stacks, sizeof(char*) * threads);
        if (stacks) s.stacks = stacks;
        if (!warps || !lanes || !fibers || !stacks) return false;
        s.cap = threads;
    }
    while (s.nfibers < threads) {
        void* m = mmap(nullptr, GUARD_BYTES + STACK_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) return false;
        if (mprotect(m, GUARD_BYTES, PROT_NONE) != 0) {
            munmap(m, GUARD_BYTES + STACK_BYTES);
            return false;
        }
        s.stacks[s.nfibers] = (char*)m;
        s.fibers[s.nfibers].started = false;
        s.nfibers++;
    }
    if (shared > s.shared_cap) {
        free(s.shared);
        s.shared = (unsigned char*)aligned_alloc(16, (shared + 15) / 16 * 16);
        s.shared_cap = s.shared ? shared : 0;
        if (!s.shared) return false;
    }
    return true;
}

int launch(void (*body)(void*), void* arg, const unsigned* shape, unsigned warp, const char** error) {
    Launch l;
    for (int d = 0; d < 3; d++) {
        l.run.grid[d] = shape[d];
        l.run.block[d] = shape[3 + d];
    }
    l.run.threads = shape[3] * shape[4] * shape[5];
    l.run.warp = warp;
    if (l.run.threads == 0 || shape[0] * shape[1] * shape[2] == 0) return 0;
    if (now) {
        *error = "a launch inside a launch";
        return 1;
    }
    if (!reserve(l.run.threads, shape[6])) {
        *error = "out of memory for the launch's fibers";
        return 1;
    }
    l.run.lanes = store.lanes;
    l.run.dynamic_shared = store.shared;
    l.fibers = store.fibers;
    l.stacks = store.stacks;
    l.wlive = store.warps;
    l.warrived = store.warps + l.run.threads;
    l.wgen = store.warps + 2 * l.run.threads;
    l.body = body;
    l.arg = arg;
    l.error = nullptr;
    now = &l;
    for (unsigned z = 0; z < l.run.grid[2]; z++)
        for (unsigned y = 0; y < l.run.grid[1]; y++)
            for (unsigned x = 0; x < l.run.grid[0]; x++) {
                l.run.bid[0] = x;
                l.run.bid[1] = y;
                l.run.bid[2] = z;
                if (!run_block(&l)) {
                    now = nullptr;
                    *error = l.error;
                    return 1;
                }
            }
    now = nullptr;
    return 0;
}

const Runtime RUNTIME = {current, block_barrier, warp_barrier, launch};

} // namespace arael_cpu

ARAEL_CPU_EXPORT const arael_cpu::Runtime* arael_cpu_runtime() { return &arael_cpu::RUNTIME; }
