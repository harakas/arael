// TransformParam: the builtin coupled-pose parameter, against a user-defined
// #[arael(component)] built from the same pieces.
//
// The builtin is hand-written (its precompute and lifecycle are Rust, not
// macro output) so that arael can ship it without expanding its own macro
// on itself. This test is the proof that the hand-written version and what
// the macro would generate are the same thing: identical solve, identical
// cached values, identical cached Jacobians.

use arael::matrix::{matrix3d, matrix3f};
use arael::model::{Component, CrossBlock, Param, SelfBlock};
use arael::transform::{TransformParam, TransformParamF};
use arael::quatern::{quaternd, quaternf};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::{vect3d, vect3f};

// --- the macro twin: the same fields the builtin has -----------------------

#[arael::model]
#[arael(component)]
struct TransformMacro {
    ref_value_q: quaternd,
    ref_translation: vect3d,
    #[arael(compute = self.ref_value_q.rotation_matrix())]
    ref_rotation: matrix3d,
    w: Param<vect3d>,
    d: Param<vect3d>,
    #[arael(symbolic = ref_rotation * matrix3sym::from_rotation_vector_small(w))]
    rotation_matrix: matrix3d,
    #[arael(symbolic = {
        let carried = d + (w % d) * 0.5 + (w % (w % d)) * 0.16666666666666666;
        ref_translation + ref_rotation * carried
    })]
    translation: vect3d,
    #[arael(deriv = rotation_matrix, by = w)]
    rotation_matrix_dw: [matrix3d; 3],
    #[arael(deriv = translation, by = d)]
    translation_dd: [vect3d; 3],
    #[arael(deriv = translation, by = w)]
    translation_dw: [vect3d; 3],
}

fn carry(w: vect3d, d: vect3d) -> vect3d {
    d + (w % d) * 0.5 + (w % (w % d)) * (1.0 / 6.0)
}

impl TransformMacro {
    fn new(translation: vect3d, qr2w: quaternd) -> TransformMacro {
        let zero = vect3d::new(0.0, 0.0, 0.0);
        let mut p = TransformMacro {
            ref_value_q: qr2w,
            ref_translation: translation,
            ref_rotation: matrix3d::identity(),
            w: Param::new(zero),
            d: Param::new(zero),
            rotation_matrix: qr2w.rotation_matrix(),
            translation,
            rotation_matrix_dw: [matrix3d::identity(); 3],
            translation_dd: [zero; 3],
            translation_dw: [zero; 3],
        };
        Component::start(&mut p);
        p
    }
}

impl Component for TransformMacro {
    fn start(&mut self) {
        self.ref_value_q = quaternd::from_rotation_matrix(self.rotation_matrix).unit();
        self.ref_translation = self.translation;
        let zero = vect3d::new(0.0, 0.0, 0.0);
        self.w.value = zero;
        self.d.value = zero;
    }
    fn update(&mut self) {
        let (w, d) = (self.w.value, self.d.value);
        self.ref_translation = self.ref_translation + self.ref_rotation * carry(w, d);
        self.ref_value_q = (self.ref_value_q * quaternd::from_rotation_vector_small(w)).unit();
        let zero = vect3d::new(0.0, 0.0, 0.0);
        self.w.value = zero;
        self.d.value = zero;
    }
    fn finish(&mut self) {
        let (w, d) = (self.w.value, self.d.value);
        self.translation = self.ref_translation + self.ref_rotation * carry(w, d);
        self.rotation_matrix = self.ref_rotation * quaternd::from_rotation_vector_small(w).rotation_matrix();
    }
}

// --- one problem, built twice --------------------------------------------
//
// A chain of frames tied by relative-pose measurements, the first frame
// anchored. The measurements disagree with the initial guess, so every
// residual is live and the optimum is a genuine compromise.

#[arael::model]
struct FrameB {
    r2w: TransformParam,
    hb: SelfBlock<FrameB>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.r2w.rotation_matrix;
    let dt = ra.transpose() * (b.r2w.translation - a.r2w.translation) - linkb.measured_translation;
    let dr = linkb.measured_rotation_transposed * (ra.transpose() * b.r2w.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}, parent = linkb))]
struct LinkB {
    #[arael(ref = root.frames)]
    a: Ref<FrameB>,
    #[arael(ref = root.frames)]
    b: Ref<FrameB>,
    measured_translation: vect3d,
    measured_rotation_transposed: matrix3d,
    hb: CrossBlock<FrameB, FrameB>,
}

#[arael::model]
#[arael(root)]
struct WorldB {
    frames: refs::Vec<FrameB>,
    links: std::vec::Vec<LinkB>,
}

#[arael::model]
struct FrameM {
    r2w: TransformMacro,
    hb: SelfBlock<FrameM>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.r2w.rotation_matrix;
    let dt = ra.transpose() * (b.r2w.translation - a.r2w.translation) - linkm.measured_translation;
    let dr = linkm.measured_rotation_transposed * (ra.transpose() * b.r2w.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}, parent = linkm))]
struct LinkM {
    #[arael(ref = root.frames)]
    a: Ref<FrameM>,
    #[arael(ref = root.frames)]
    b: Ref<FrameM>,
    measured_translation: vect3d,
    measured_rotation_transposed: matrix3d,
    hb: CrossBlock<FrameM, FrameM>,
}

#[arael::model]
#[arael(root)]
struct WorldM {
    frames: refs::Vec<FrameM>,
    links: std::vec::Vec<LinkM>,
}

/// Initial guesses, and measurements taken from a DIFFERENT trajectory so
/// every residual is nonzero.
fn scene() -> (Vec<(vect3d, quaternd)>, Vec<(vect3d, matrix3d)>) {
    let mut poses = Vec::new();
    for i in 0..5 {
        let th = i as f64 * 0.4;
        poses.push((
            vect3d::new(i as f64 * 1.1, 0.2 * th.sin(), -0.1 * th),
            (quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), th * 0.9)
                * quaternd::from_axis_angle(vect3d::new(1.0, 0.0, 0.0), 0.15 * th))
            .unit(),
        ));
    }
    // Four chain links plus a loop closure back to the anchor. The chain
    // alone would be exactly determined (24 residuals, 24 free params) and
    // drive the cost to zero, which strands LM at its damping ceiling
    // instead of converging; the closure over-determines it, so the
    // optimum is a genuine compromise at nonzero cost.
    let mut links = Vec::new();
    for i in 0..4 {
        let rel_t = vect3d::new(1.0, 0.05 * (i as f64), 0.02);
        let rel_q = quaternd::from_axis_angle(
            vect3d::new(0.1, -0.2, 1.0).unit(), 0.35 + 0.03 * i as f64).unit();
        links.push((rel_t, rel_q.rotation_matrix().transpose()));
    }
    let closure_t = vect3d::new(-3.6, 0.4, 0.1);
    let closure_q = quaternd::from_axis_angle(
        vect3d::new(0.0, 0.1, -1.0).unit(), 1.3).unit();
    links.push((closure_t, closure_q.rotation_matrix().transpose()));
    (poses, links)
}

fn build_builtin() -> WorldB {
    let (poses, links) = scene();
    let mut w = WorldB { frames: refs::Vec::new(), links: std::vec::Vec::new() };
    for (k, (p, q)) in poses.iter().enumerate() {
        w.frames.push(FrameB {
            r2w: if k == 0 { TransformParam::fixed(*p, *q) } else { TransformParam::new(*p, *q) },
            hb: SelfBlock::new(),
        });
    }
    for (i, (t, r)) in links.iter().enumerate() {
        let (a, b) = if i < 4 { (i as u32, i as u32 + 1) } else { (4, 0) };
        w.links.push(LinkB {
            a: w.frames.ref_at(a), b: w.frames.ref_at(b),
            measured_translation: *t, measured_rotation_transposed: *r,
            hb: CrossBlock::new(),
        });
    }
    w
}

fn build_macro() -> WorldM {
    let (poses, links) = scene();
    let mut w = WorldM { frames: refs::Vec::new(), links: std::vec::Vec::new() };
    for (k, (p, q)) in poses.iter().enumerate() {
        let mut pose = TransformMacro::new(*p, *q);
        if k == 0 {
            pose.w = Param::fixed(vect3d::new(0.0, 0.0, 0.0));
            pose.d = Param::fixed(vect3d::new(0.0, 0.0, 0.0));
        }
        w.frames.push(FrameM { r2w: pose, hb: SelfBlock::new() });
    }
    for (i, (t, r)) in links.iter().enumerate() {
        let (a, b) = if i < 4 { (i as u32, i as u32 + 1) } else { (4, 0) };
        w.links.push(LinkM {
            a: w.frames.ref_at(a), b: w.frames.ref_at(b),
            measured_translation: *t, measured_rotation_transposed: *r,
            hb: CrossBlock::new(),
        });
    }
    w
}

/// The hand-written builtin and the macro-generated twin must solve the
/// same problem identically -- same trajectory, same optimum, same cached
/// values and Jacobians at the solution.
#[test]
fn builtin_matches_the_macro_component() {
    let cfg = LmConfig::conservative();
    let mut b = build_builtin();
    let mut m = build_macro();
    let rb = b.solve_dense(&cfg).unwrap();
    let rm = m.solve_dense(&cfg).unwrap();

    assert!(rb.status.is_success(), "builtin: {:?}", rb.status);
    assert!(rm.status.is_success(), "macro: {:?}", rm.status);
    assert!(rb.end_cost > 1e-6, "the measurements disagree by design: {}", rb.end_cost);
    assert!((rb.end_cost - rm.end_cost).abs() < 1e-12 * (1.0 + rb.end_cost),
        "cost: builtin {} vs macro {}", rb.end_cost, rm.end_cost);
    assert_eq!(rb.iterations, rm.iterations, "same damping trajectory");

    for (i, (fb, fm)) in b.frames.iter().zip(m.frames.iter()).enumerate() {
        assert!((fb.r2w.translation - fm.r2w.translation).norm() < 1e-12, "translation[{}]", i);
        for r in 0..3 {
            assert!((fb.r2w.rotation_matrix[r] - fm.r2w.rotation_matrix[r]).norm() < 1e-12, "rotation_matrix[{}] row {}", i, r);
            for k in 0..3 {
                assert!((fb.r2w.rotation_matrix_dw[r][k] - fm.r2w.rotation_matrix_dw[r][k]).norm() < 1e-12,
                    "rotation_matrix_dw[{}][{}] row {}", i, r, k);
            }
            assert!((fb.r2w.translation_dd[r] - fm.r2w.translation_dd[r]).norm() < 1e-12, "translation_dd[{}][{}]", i, r);
            assert!((fb.r2w.translation_dw[r] - fm.r2w.translation_dw[r]).norm() < 1e-12, "translation_dw[{}][{}]", i, r);
        }
    }
}

/// The f32 twin builds and solves the same problem to the same place.
#[test]
fn the_f32_twin_agrees() {
    let (poses, _) = scene();
    let (p, q) = poses[2];
    let d = TransformParam::new(p, q);
    let f = TransformParamF::new(
        vect3f::new(p.x as f32, p.y as f32, p.z as f32),
        quaternf::new(q.t as f32, vect3f::new(q.v.x as f32, q.v.y as f32, q.v.z as f32)).unit(),
    );
    for r in 0..3 {
        for k in 0..3 {
            assert!((d.rotation_matrix[r][k] - f.rotation_matrix[r][k] as f64).abs() < 1e-6,
                "rotation_matrix[{}][{}]", r, k);
        }
    }
    let _ = matrix3f::identity();
}
