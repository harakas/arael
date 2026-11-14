#!/bin/bash
# The cases again from the traces run.sh left (DEVMAP_TRACES), without running the benches.
# The probed cases first: a GPU's case takes its access patterns from the
# largest probed case of the same bench and model (donor.sh).
here=$(cd "$(dirname "$0")" && pwd)
out=${DEVMAP_TRACES:-/tmp/devmap-traces}
for pass in cpu gpu; do
    for t in "$out"/*.jsonl; do
        id=$(basename "$t" .jsonl)
        IFS=- read -r bench size model rt <<< "$id"
        if [ "$rt" = cpu ]; then [ $pass = cpu ] || continue; else [ $pass = gpu ] || continue; fi
        case $bench in bal) problem="bal Ladybug-$size" ;; slam) problem="slam $size poses" ;; *) problem=$bench ;; esac
        where=$rt
        from=()
        if [ "$rt" = cpu ]; then
            where="CPU device (probed)"
        else
            f=$("$here/donor.sh" "$bench" "$model")
            [ -n "$f" ] && from=(--patterns-from "$f")
        fi
        python3 "$here/devmap.py" "$t" --id "$id" --title "$problem, $model model, $where" --problem "$problem" --model "$model" "${from[@]}"
    done
done
