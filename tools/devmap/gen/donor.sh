#!/bin/bash
# The probed case a GPU's case borrows its access patterns from: the case
# file of the largest probed problem of a bench and model, or nothing.
# Usage: gen/donor.sh bal f32   ->   .../data/bal-138-f32-cpu.js
here=$(cd "$(dirname "$0")" && pwd)
best= max=-1
for f in "$here/../data/$1"-*-"$2"-cpu.js; do
    [ -f "$f" ] || continue
    n=$(basename "$f" | cut -d- -f2)
    if [ "$n" -gt "$max" ] 2>/dev/null; then
        max=$n
        best=$f
    fi
done
echo "$best"
