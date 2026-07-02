#!/bin/bash
# Feature-matrix build check for the arael crate.
#
# The optional solver backends (lapack, eigen, cholmod) are behind cargo
# features and are NOT built by a plain `cargo test`. This script compiles
# and tests each feature combination so feature-gated code cannot silently
# rot (see REVIEW.md B1: all three features were uncompilable for ~118
# commits because nothing ever built them).
#
# Requirements: system LAPACK (liblapack), Eigen3 headers, SuiteSparse/
# CHOLMOD for the respective features. Run before every release.

set -e
cd "$(dirname "$0")/.."

run() {
    echo "=== $* ==="
    "$@"
}

run cargo test -p arael --lib --no-default-features
run cargo test -p arael --lib
run cargo test -p arael --lib --features lapack
run cargo test -p arael --lib --features eigen
run cargo test -p arael --lib --features cholmod
run cargo check -p arael --tests --examples --features lapack,cholmod

echo "=== all feature combinations OK ==="
