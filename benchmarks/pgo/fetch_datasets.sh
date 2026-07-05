#!/bin/sh
# Canonical 2D pose-graph benchmark datasets.
#   M3500     -- Olson's Manhattan world (3500 poses, 5453 edges), the file
#                shipped by tiny-solver-rs, matching Carlone's
#                input_M3500_g2o naming; diagonal information matrices and
#                odometry initialization included.
#   city10000 -- the iSAM city dataset (10000 poses, 20687 edges), from
#                the SE-Sync data collection; diagonal information
#                (50, 50, 100) and odometry initialization included.
set -e
cd "$(dirname "$0")/datasets"
[ -f input_M3500_g2o.g2o ] || curl -sL -o input_M3500_g2o.g2o \
  "https://raw.githubusercontent.com/powei-lin/tiny-solver-rs/master/tests/data/input_M3500_g2o.g2o"
[ -f city10000.g2o ] || curl -sL -o city10000.g2o \
  "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/city10000.g2o"
ls -la
