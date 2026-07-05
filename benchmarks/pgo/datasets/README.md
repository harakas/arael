# Benchmark datasets

Vendored copies of the two canonical 2D pose-graph benchmark datasets.
Both have circulated freely across SLAM solver repositories for nearly
two decades (g2o, GTSAM, Ceres, SE-Sync, tiny-solver, ...) as community
benchmark artifacts; neither carries a dedicated data license upstream.

- `input_M3500_g2o.g2o` -- Olson's Manhattan world (M3500).
  Cite: E. Olson, J. Leonard, S. Teller, "Fast iterative alignment of
  pose graphs with poor initial estimates", ICRA 2006.
  This copy: tiny-solver-rs (github.com/powei-lin/tiny-solver-rs,
  MIT license), tests/data/input_M3500_g2o.g2o -- the revision from
  Luca Carlone's 2D pose-graph dataset collection.

- `city10000.g2o` -- the iSAM city dataset (City10000).
  Cite: M. Kaess, A. Ranganathan, F. Dellaert, "iSAM: Incremental
  Smoothing and Mapping", IEEE TRO 2008.
  This copy: SE-Sync (github.com/david-m-rosen/SE-Sync, LGPL-3
  repository; the data directory carries no separate license),
  data/city10000.g2o.

`../fetch_datasets.sh` re-downloads both from the same sources.
