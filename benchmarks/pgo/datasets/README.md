# Benchmark datasets

Vendored copies of the canonical 2D and 3D pose-graph benchmark
datasets. All have circulated freely across SLAM solver repositories
for nearly two decades (g2o, GTSAM, Ceres, SE-Sync, tiny-solver, ...)
as community benchmark artifacts; none carries a dedicated data
license upstream.

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

- `sphere2500.g2o` -- the synthetic sphere benchmark (2500 poses,
  4949 SE3 edges), originally released with iSAM and distributed with
  GTSAM as sphere2500.
  Cite: L. Carlone, R. Tron, K. Daniilidis, F. Dellaert,
  "Initialization Techniques for 3D SLAM: a Survey on Rotation
  Estimation and its Use in Pose Graph Optimization", ICRA 2015 --
  the citation recommended by the 3D dataset collection page
  (lucacarlone.mit.edu/datasets).
  This copy: SE-Sync (github.com/david-m-rosen/SE-Sync),
  data/sphere2500.g2o.

- `parking-garage.g2o` -- real-world multi-level parking garage
  dataset (1661 poses, 6275 SE3 edges).
  Cite: same Carlone et al. ICRA 2015 survey (the collection paper
  for the standard 3D pose-graph benchmarks).
  This copy: SE-Sync (github.com/david-m-rosen/SE-Sync),
  data/parking-garage.g2o.

`../fetch_datasets.sh` re-downloads all of them from the same sources.
