// Generic entities and constraint structs: one model over `T: Float`,
// instantiated by two concrete roots (f64 and f32). Covers a
// self-constraint, a cross-constraint with `Ref` fields and a
// `CrossBlock<A<T>, A<T>, T>`, the remote-block (localization) shape, a
// built-in component field at `<T>`, and bare `T` data fields read from
// constraint bodies.

use arael::model::{CrossBlock, Model, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::unitvec::UnitVecParam;
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d, vect3f};

// ------------------------------------------------------------------- model

/// A point pulled toward its own `(tx, ty)` prior.
#[arael::model]
#[arael(constraint(hb, {
    [(pt.pos.x - pt.tx) * 0.3, (pt.pos.y - pt.ty) * 0.3]
}))]
struct Pt<T: Float> {
    pos: Param<vect2<T>>,
    tx: T,
    ty: T,
    hb: SelfBlock<Pt<T>, T>,
}

/// A spring between two points with rest length `rest`.
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(sqrt(dx * dx + dy * dy) - link.rest) * link.w]
}, parent = link))]
struct Link<T: Float> {
    #[arael(ref = root.pts)]
    a: Ref<Pt<T>>,
    #[arael(ref = root.pts)]
    b: Ref<Pt<T>>,
    rest: T,
    w: T,
    hb: CrossBlock<Pt<T>, Pt<T>, T>,
}

/// A unit-direction landmark (built-in component at `<T>`) carrying
/// nested cross-observations.
#[arael::model]
struct Lm<T: Float> {
    dir: UnitVecParam<T>,
    obs: std::vec::Vec<LmObs<T>>,
    hb: SelfBlock<Lm<T>, T>,
}

/// Nested cross-observation: lives in a Vec ON the landmark and couples
/// it to a point through a CrossBlock. The block's per-instantiation
/// Model bound must propagate through `Vec<LmObs<T>>` into `Lm<T>`'s
/// generated impl -- the owner names the collection, not the block.
#[arael::model]
#[arael(constraint(hb, parent = lm, {
    [(lm.dir.unit.x * pt.pos.x + lm.dir.unit.y * pt.pos.y - lmobs.dot) * 0.01]
}))]
struct LmObs<T: Float> {
    #[arael(ref = root.pts)]
    pt: Ref<Pt<T>>,
    dot: T,
    hb: CrossBlock<Lm<T>, Pt<T>, T>,
}

/// Direction observation writing to the landmark's own block (the
/// remote-block localization shape).
#[arael::model]
#[arael(constraint(lm.hb, {
    [lm.dir.unit.x - dirobs.m.x, lm.dir.unit.y - dirobs.m.y, lm.dir.unit.z - dirobs.m.z]
}))]
struct DirObs<T: Float> {
    #[arael(ref = root.lms)]
    lm: Ref<Lm<T>>,
    m: vect3<T>,
}

#[arael::model]
#[arael(root)]
struct World64 {
    pts: refs::Vec<Pt<f64>>,
    lms: refs::Vec<Lm<f64>>,
    links: std::vec::Vec<Link<f64>>,
    dobs: std::vec::Vec<DirObs<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct World32 {
    pts: refs::Vec<Pt<f32>>,
    lms: refs::Vec<Lm<f32>>,
    links: std::vec::Vec<Link<f32>>,
    dobs: std::vec::Vec<DirObs<f32>>,
}

// ------------------------------------------------------------------- scene

const N: usize = 6;

fn prior(i: usize) -> (f64, f64) {
    (i as f64, 0.5 * (i % 2) as f64)
}

fn measured_dir(k: usize) -> vect3d {
    vect3d::new(0.2, 1.0 + k as f64, -0.5 * k as f64).unit()
}

macro_rules! build_world {
    ($world:ident, $t:ty) => {{
        let mut w = $world {
            pts: refs::Vec::new(),
            lms: refs::Vec::new(),
            links: std::vec::Vec::new(),
            dobs: std::vec::Vec::new(),
        };
        let mut pt_refs = std::vec::Vec::new();
        for i in 0..N {
            let (tx, ty) = prior(i);
            pt_refs.push(w.pts.push(Pt {
                pos: Param::new(vect2::new((tx + 0.3) as $t, (ty - 0.2) as $t)),
                tx: tx as $t,
                ty: ty as $t,
                hb: SelfBlock::new(),
            }));
        }
        for i in 0..N - 1 {
            w.links.push(Link {
                a: pt_refs[i],
                b: pt_refs[i + 1],
                rest: 0.8 as $t,
                w: 2.0 as $t,
                hb: CrossBlock::new(),
            });
        }
        for k in 0..2 {
            let m = measured_dir(k);
            // A weak nested cross-observation per landmark: dot of the
            // direction's xy with the point's prior position.
            let (tx, ty) = prior(k + 1);
            let lm = w.lms.push(Lm {
                dir: UnitVecParam::new(vect3::new(1.0 as $t, 0.1 as $t, 0.0 as $t)),
                obs: vec![LmObs {
                    pt: pt_refs[k + 1],
                    dot: (m.x * tx + m.y * ty) as $t,
                    hb: CrossBlock::new(),
                }],
                hb: SelfBlock::new(),
            });
            w.dobs.push(DirObs {
                lm,
                m: vect3::new(m.x as $t, m.y as $t, m.z as $t),
            });
        }
        w
    }};
}

// ------------------------------------------------------------------- tests

/// Params fold correctly at both precisions; data-only constraint structs
/// carry none.
#[test]
fn generic_entity_param_counts() {
    assert_eq!(<Pt<f64> as Model>::PARAM_COUNT, 2);
    assert_eq!(<Pt<f32> as Model>::PARAM_COUNT, 2);
    assert_eq!(<Lm<f64> as Model>::PARAM_COUNT, 2);
    assert_eq!(<Link<f64> as Model>::PARAM_COUNT, 0);
}

/// The same generic model solved by an f64 root and an f32 root lands on
/// the same optimum.
#[test]
fn f64_and_f32_roots_agree() {
    // The priors and springs genuinely conflict, so the optimum is a
    // nonzero equilibrium (~0.081); both precisions must find it.
    let mut w64 = build_world!(World64, f64);
    let r64 = w64.solve_sparse(&LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    assert!(r64.end_cost < 0.1, "f64 cost {}", r64.end_cost);

    let mut w32 = build_world!(World32, f32);
    let r32 = w32.solve_sparse(&LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    assert!((r32.end_cost as f64 - r64.end_cost).abs() < 1e-3,
        "f32 cost {} vs f64 {}", r32.end_cost, r64.end_cost);

    for (i, (p64, p32)) in w64.pts.iter().zip(w32.pts.iter()).enumerate() {
        let dx = p64.pos.value.x - p32.pos.value.x as f64;
        let dy = p64.pos.value.y - p32.pos.value.y as f64;
        assert!(dx.abs() < 2e-3 && dy.abs() < 2e-3,
            "pt {i}: f64 ({:.6},{:.6}) vs f32 ({:.6},{:.6})",
            p64.pos.value.x, p64.pos.value.y, p32.pos.value.x, p32.pos.value.y);
    }

    // The remote-block observations drive each landmark near its measured
    // direction (the weak nested cross-observation tugs it slightly), and
    // the two precisions agree.
    for (k, (l64, l32)) in w64.lms.iter().zip(w32.lms.iter()).enumerate() {
        let m = measured_dir(k);
        assert!((l64.dir.unit - m).norm() < 1e-2, "lm {k} f64 off target");
        let u32v = vect3f::new(l32.dir.unit.x, l32.dir.unit.y, l32.dir.unit.z);
        let u64v = vect3f::new(l64.dir.unit.x as f32, l64.dir.unit.y as f32,
            l64.dir.unit.z as f32);
        assert!((u32v - u64v).norm() < 2e-3, "lm {k} f32 vs f64 disagree");
    }
}
