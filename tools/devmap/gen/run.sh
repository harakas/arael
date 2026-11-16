#!/bin/bash
# Traces the bal and slam benchmarks' device solves and turns them into the
# viewer's cases (../data). Usage: gen/run.sh [cpu] [cuda] [hip] (default: cpu cuda).
#
# cpu:  the CPU device with its access probe (clang): small problems, every
#       read and write measured. BAL_CPU_SIZES (default "49 138") and
#       SLAM_CPU_POSES (default 120) pick the problems, empty for none;
#       BAL_CPU_ATTEMPT_SIZES (default none) the bal problems probed for one
#       attempt (BENCH_SOLVE_ONLY=1 BENCH_MAX_ITERS=1), too large to probe a
#       whole solve.
# cuda, hip: the GPU at the benchmark's size: the launch sequence and the
#       buffers, reads and writes from the kernels' signatures.
#
# Needs the benches built with the trace:
#   (cd benchmarks/bal  && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
#   (cd benchmarks/slam && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${DEVMAP_TRACES:-/tmp/devmap-traces}
mkdir -p "$out"
[ $# -eq 0 ] && set -- cpu cuda

trace() { # runtime bench id title problem model env...
    local rt=$1 bench=$2 id=$3 title=$4 problem=$5 model=$6
    shift 6
    local bin=bal-bench
    [ "$bench" = slam ] && bin=slam-bench
    echo "== $id"
    # regen.sh takes the title from here
    echo "$title" > "$out/$id.title"
    # a GPU's launches are timed (one at a time); under the probe a time says nothing
    local timed=0
    [ "$rt" != cpu ] && timed=1
    (cd "$repo/benchmarks/$bench" && env "$@" ARAEL_GPU=$rt ARAEL_GPU_TRACE="$out/$id.jsonl" ARAEL_GPU_TRACE_TIME=$timed ROUNDS=1 BENCH_THREADS=8 BENCH_PIN=0 \
        ./target-trace/release/$bin > "$out/$id.log" 2>&1) || { tail -5 "$out/$id.log"; return 1; }
    # a GPU's case takes its access patterns from the largest probed case of the same bench and model
    local from=() f
    if [ "$rt" != cpu ]; then
        f=$("$here/donor.sh" "$bench" "$model")
        [ -n "$f" ] && from=(--patterns-from "$f")
    fi
    python3 "$here/devmap.py" "$out/$id.jsonl" --id "$id" --title "$title" --problem "$problem" --model "$model" "${from[@]}"
}

for rt in "$@"; do
    case $rt in
    cpu)
        export ARAEL_CPU_PROBE=1
        # Ladybug-49 takes about 1.5 minutes probed, Ladybug-138 about 5: the larger one
        # launches factor kernels the smaller does not (k_pair_update, k_small_factor, k_bwd_gemv)
        for n in ${BAL_CPU_SIZES-49 138}; do
            trace cpu bal bal-$n-f32-cpu "bal Ladybug-$n, f32 model, CPU device (probed)" "bal Ladybug-$n" f32 BAL_ONLY=$n "BAL_SYSTEMS=arael LM f32 schur" BAL_NO_MEM=1
            trace cpu bal bal-$n-f64-cpu "bal Ladybug-$n, f64 model, CPU device (probed)" "bal Ladybug-$n" f64 BAL_ONLY=$n "BAL_SYSTEMS=arael LM f64 schur" BAL_NO_MEM=1
        done
        # one attempt: the factor kernels a smaller problem does not launch
        for n in ${BAL_CPU_ATTEMPT_SIZES-}; do
            for m in f32 f64; do
                trace cpu bal bal-$n-$m-cpu "bal Ladybug-$n, $m model, CPU device (probed, one attempt)" "bal Ladybug-$n" $m BAL_ONLY=$n "BAL_SYSTEMS=arael LM $m schur\$" BAL_NO_MEM=1 BENCH_SOLVE_ONLY=1 BENCH_MAX_ITERS=1
            done
        done
        for n in ${SLAM_CPU_POSES-120}; do
            trace cpu slam slam-$n-f32-cpu "slam $n poses, f32 model, CPU device (probed)" "slam $n poses" f32 SLAM_POSES=$n "SLAM_SYSTEMS=arael LM f32" SLAM_NO_MEM=1
            trace cpu slam slam-$n-f64-cpu "slam $n poses, f64 model, CPU device (probed)" "slam $n poses" f64 SLAM_POSES=$n "SLAM_SYSTEMS=arael LM f64" SLAM_NO_MEM=1
        done
        unset ARAEL_CPU_PROBE
        ;;
    cuda | hip)
        [ $rt = cuda ] && export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        [ $rt = hip ] && export AMD_SERIALIZE_COPY=3 AMD_SERIALIZE_KERNEL=3
        trace $rt bal bal-372-f32-$rt "bal Ladybug-372, f32 model, $rt" "bal Ladybug-372" f32 BAL_ONLY=372 "BAL_SYSTEMS=arael LM f32 schur" BAL_NO_MEM=1
        trace $rt bal bal-372-f64-$rt "bal Ladybug-372, f64 model, $rt" "bal Ladybug-372" f64 BAL_ONLY=372 "BAL_SYSTEMS=arael LM f64 schur" BAL_NO_MEM=1
        trace $rt slam slam-900-f32-$rt "slam 900 poses, f32 model, $rt" "slam 900 poses" f32 SLAM_POSES=900 "SLAM_SYSTEMS=arael LM f32" SLAM_NO_MEM=1
        trace $rt slam slam-900-f64-$rt "slam 900 poses, f64 model, $rt" "slam 900 poses" f64 SLAM_POSES=900 "SLAM_SYSTEMS=arael LM f64" SLAM_NO_MEM=1
        ;;
    *) echo "unknown runtime $rt (cpu, cuda, hip)"; exit 1 ;;
    esac
done
