#!/bin/sh
# BAL bundle-adjustment benchmark datasets (see datasets/README.md).
set -e
cd "$(dirname "$0")/datasets"
[ -f problem-49-7776-pre.txt ] || {
  curl -sL -o problem-49-7776-pre.txt.bz2 \
    "https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2"
  bunzip2 problem-49-7776-pre.txt.bz2
}
ls -la
