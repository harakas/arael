//! Stage 3 of nested-model-tree support: nested CROSS constraints and
//! cross-LEVEL references. A minimal 1D multi-path SLAM:
//!
//!   Map { paths: Vec<Path>, anchors: refs::Vec<Anchor> }   (root)
//!   Path { nodes, links, obs }                             (block-less sub-model)
//!   Node   -- passive entity, lives in Path.nodes (two hops from root)
//!   Anchor -- passive shared entity, lives in Map.anchors (one hop)
//!   Link -- cross constraint in Path.links: chains two nodes of THE SAME path
//!           (`parent.nodes`), a 1D odometry delta.
//!   Obs  -- cross-LEVEL constraint in Path.obs: ties a node (`parent.nodes`)
//!           to a shared anchor (`root.anchors`) -- the merge point.
//!
//! This exercises: a constraint struct two hops down, `parent.`-relative refs
//! (resolve against the containing Path), `root.`-relative refs (resolve
//! against the Map), a cross block coupling a nested entity with a root entity,
//! and passive wiring of nested Nodes (no self-constraint). The anchors are
//! held fixed, pinning the gauge; with exact measurements every node must
//! recover its true position -- which it can only do if all of the above wire
//! correctly.

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmConfig;
use arael::simple_lm::LmProblem;

#[arael::model]
struct Node {
    x: Param<f32>,
    hb: SelfBlock<Node, f32>,
}

#[arael::model]
struct Anchor {
    x: Param<f32>,
    hb: SelfBlock<Anchor, f32>,
}

// Chain: cur.x - prev.x must equal the measured delta. Both nodes are in the
// SAME path -> `parent.nodes`.
#[arael::model]
#[arael(constraint(hb, {
    [(cur.x - prev.x - link.measured) * link.isigma]
}))]
struct Link {
    #[arael(ref = parent.nodes)] prev: Ref<Node>,
    #[arael(ref = parent.nodes)] cur: Ref<Node>,
    measured: f32,
    isigma: f32,
    hb: CrossBlock<Node, Node, f32>,
}

// Observation: anchor.x - node.x must equal the measured offset. node is in the
// containing path (`parent.nodes`); anchor is the shared map (`root.anchors`).
#[arael::model]
#[arael(constraint(hb, {
    [(anchor.x - node.x - obs.measured) * obs.isigma]
}))]
struct Obs {
    #[arael(ref = parent.nodes)] node: Ref<Node>,
    #[arael(ref = root.anchors)] anchor: Ref<Anchor>,
    measured: f32,
    isigma: f32,
    hb: CrossBlock<Anchor, Node, f32>,
}

#[arael::model]
struct Path {
    nodes: refs::Vec<Node>,
    links: std::vec::Vec<Link>,
    obs: std::vec::Vec<Obs>,
}

#[arael::model]
#[arael(root, f32)]
struct Map {
    paths: std::vec::Vec<Path>,
    anchors: refs::Vec<Anchor>,
}

// Build one path: nodes initialized at 0, chained by exact deltas, with an
// observation of `anchor_idx` from node `obs_node`. `truth` is the true node
// positions used to synthesize exact measurements.
fn build_path(truth: &[f32], anchors_truth: &[f32], obs: &[(usize, usize)],
              anchor_refs: &[Ref<Anchor>]) -> Path {
    let mut path = Path {
        nodes: refs::Vec::new(),
        links: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for _ in truth { path.nodes.push(Node { x: Param::new(0.0), hb: SelfBlock::new() }); }
    for i in 1..truth.len() {
        path.links.push(Link {
            prev: path.nodes.ref_at(i - 1),
            cur: path.nodes.ref_at(i),
            measured: truth[i] - truth[i - 1],
            isigma: 1.0,
            hb: CrossBlock::new(),
        });
    }
    for &(node_i, anchor_i) in obs {
        path.obs.push(Obs {
            node: path.nodes.ref_at(node_i),
            anchor: anchor_refs[anchor_i],
            measured: anchors_truth[anchor_i] - truth[node_i],
            isigma: 1.0,
            hb: CrossBlock::new(),
        });
    }
    path
}

fn build_map() -> (Map, Vec<Vec<f32>>) {
    // Two shared anchors, held fixed -- the common reference frame both paths
    // are pulled into.
    let anchors_truth = [0.0_f32, 10.0];
    let mut anchors = refs::Vec::new();
    let mut anchor_refs = Vec::new();
    for &a in &anchors_truth {
        let mut anc = Anchor { x: Param::new(a), hb: SelfBlock::new() };
        anc.x.optimize = false; // fixed reference
        anchor_refs.push(anchors.push(anc));
    }

    // Two paths with different true node positions, each observing both anchors
    // from different nodes -- so the shared anchors merge the two runs.
    let path_truth = vec![vec![1.0_f32, 2.0, 3.0], vec![1.5_f32, 4.0, 6.5]];
    let paths = vec![
        build_path(&path_truth[0], &anchors_truth, &[(0, 0), (2, 1)], &anchor_refs),
        build_path(&path_truth[1], &anchors_truth, &[(0, 0), (2, 1)], &anchor_refs),
    ];

    (Map { paths, anchors }, path_truth)
}

#[test]
fn nested_cross_and_cross_level_converge_dense() {
    let (mut map, truth) = build_map();
    let cfg = LmConfig::<f32>::default();
    let result = map.solve_dense(&cfg).unwrap();
    assert!(result.end_cost < 1e-6, "did not converge: cost {}", result.end_cost);

    for (p, t) in map.paths.iter().zip(truth.iter()) {
        for (n, &tx) in p.nodes.iter().zip(t.iter()) {
            assert!((n.x.value - tx).abs() < 1e-3,
                "node x={} did not reach truth {}", n.x.value, tx);
        }
    }
    // Anchors held fixed.
    assert_eq!(map.anchors[0].x.value, 0.0);
    assert_eq!(map.anchors[1].x.value, 10.0);
}

#[test]
fn nested_cross_and_cross_level_converge_sparse() {
    let (mut map, truth) = build_map();
    let cfg = LmConfig::<f32>::default();
    let result = map.solve_sparse(&cfg).unwrap();
    assert!(result.end_cost < 1e-6, "did not converge: cost {}", result.end_cost);

    for (p, t) in map.paths.iter().zip(truth.iter()) {
        for (n, &tx) in p.nodes.iter().zip(t.iter()) {
            assert!((n.x.value - tx).abs() < 1e-3,
                "node x={} did not reach truth {}", n.x.value, tx);
        }
    }
}
