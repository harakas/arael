// The verbose log's line of an attempt, `i/j: a->b / d, lambda=l (step=us
// ...)`: what the linear solver adds to it (`LmSolver::attempt_note`), and
// that it writes no line of its own beside it. One test in a binary of its
// own: the log's sink is the process's.

use std::sync::{Arc, Mutex};

use arael::log;
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{lm_solve, BlockSupernodalMode, EnvelopeMode, LmConfig, MixedPrecision, RootProblem, SchurPolicy, SparseFaer, SparseFaerOptions};

#[arael::model]
#[arael(constraint(hb, { [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1] }))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Landmark {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Landmark>,
}

#[arael::model]
#[arael(constraint(hb, { [l.x - p.x - obs.dx, l.y - p.y - obs.dy] }))]
struct Obs {
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Landmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    obs: std::vec::Vec<Obs>,
}

/// Six poses near their anchors, each seeing eight landmarks.
fn build() -> World {
    let mut w = World { poses: refs::Vec::new(), landmarks: refs::Vec::new(), obs: std::vec::Vec::new() };
    let pose = |i: usize| (i as f64, 0.5 * i as f64);
    let landmark = |j: usize| (j as f64, 2.0 + j as f64);
    for i in 0..6 {
        let (x, y) = pose(i);
        w.poses.push(Pose { x: Param::new(x + 0.05), y: Param::new(y - 0.05), ax: x, ay: y, hb: SelfBlock::new() });
    }
    for j in 0..8 {
        let (x, y) = landmark(j);
        w.landmarks.push(Landmark { x: Param::new(x - 0.05), y: Param::new(y + 0.05), hb: SelfBlock::new() });
    }
    for i in 0..6 {
        for j in 0..8 {
            let ((px, py), (lx, ly)) = (pose(i), landmark(j));
            w.obs.push(Obs { p: w.poses.ref_at(i), l: w.landmarks.ref_at(j), dx: lx - px, dy: ly - py, hb: CrossBlock::new() });
        }
    }
    w
}

/// The `name=value` items of a note, each value in microseconds: their
/// names. Panics on an item of another form.
fn timed(items: &[&str]) -> Vec<String> {
    items
        .iter()
        .map(|item| {
            let (name, us) = item.split_once('=').unwrap_or_else(|| panic!("`{item}` is not name=value"));
            assert!(us.parse::<u64>().is_ok(), "`{item}`: microseconds");
            name.to_string()
        })
        .collect()
}

#[test]
fn an_attempts_line_carries_what_the_linear_solver_adds() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    log::set_sink(move |_, msg| sink.lock().unwrap().push(msg.to_string()));
    // a verbose solve's attempts: what follows `step=us` on each one's line
    let solve = |opts: SparseFaerOptions| -> Vec<Vec<String>> {
        seen.lock().unwrap().clear();
        let mut w = build();
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        lm_solve(&params, &mut SparseFaer::from_options(&opts), &mut w, &LmConfig { max_iters: 20, verbose: true, ..Default::default() }).unwrap();
        let log = seen.lock().unwrap().clone();
        // the linear solver writes no line of its own an attempt
        assert!(!log.iter().any(|m| m.starts_with("supernodal:") || m.starts_with("mixed precision:")), "{log:#?}");
        let notes: Vec<Vec<String>> = log
            .iter()
            .filter_map(|m| m.rsplit_once(" (step="))
            .map(|(_, tail)| {
                let tail = tail.strip_suffix(')').unwrap_or_else(|| panic!("`{tail}` ends the line"));
                let mut items = tail.split(", ").map(str::to_string);
                assert!(items.next().unwrap().parse::<u64>().is_ok(), "`{tail}`: the step's microseconds");
                items.collect()
            })
            .collect();
        assert!(!notes.is_empty(), "no attempt in {log:#?}");
        notes
    };
    let reduced = || SparseFaerOptions::default().with_policy(SchurPolicy::Force);
    let supernodal = || reduced().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Always);
    fn refs(note: &[String]) -> Vec<&str> {
        note.iter().map(String::as_str).collect()
    }

    // the block supernodal factor: where its time went
    for note in solve(supernodal()) {
        assert_eq!(timed(&refs(&note)), ["factor", "solve"], "{note:?}");
    }
    // the same for the whole system in block form
    for note in solve(SparseFaerOptions::default().with_policy(SchurPolicy::Never).with_block_supernodal(BlockSupernodalMode::Always)) {
        assert_eq!(timed(&refs(&note)), ["factor", "solve"], "{note:?}");
    }
    // mixed precision: the refinement's solves and residuals, its steps
    // and the residual it reached
    for note in solve(supernodal().with_mixed_precision(Some(MixedPrecision::default()))) {
        let note = refs(&note);
        assert_eq!(note.len(), 5, "{note:?}");
        assert_eq!(timed(&note[..3]), ["factor", "solves", "residuals"], "{note:?}");
        let steps = note[3].strip_suffix(" refinement steps").or_else(|| note[3].strip_suffix(" refinement step")).unwrap_or_else(|| panic!("{note:?}"));
        assert!(steps.parse::<usize>().is_ok(), "{note:?}");
        let residual = note[4].strip_prefix("residual ").unwrap_or_else(|| panic!("{note:?}"));
        let residual = residual.strip_suffix(" short of tolerance").unwrap_or(residual);
        assert!(residual.parse::<f64>().is_ok_and(|r| r < 1e-6), "{note:?}");
    }
    // mixed precision without a refinement step
    for note in solve(supernodal().with_mixed_precision(Some(MixedPrecision { max_refinements: 0, ..Default::default() }))) {
        let note = refs(&note);
        assert_eq!((timed(&note[..2]), note[2]), (vec!["factor".to_string(), "solve".to_string()], "no refinement"), "{note:?}");
    }
    // a route that adds nothing: the scalar sparse Cholesky
    for note in solve(reduced().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Never)) {
        assert!(note.is_empty(), "{note:?}");
    }
    log::reset_sink();
}
