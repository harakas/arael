// A model crate: entities, a component, a constraint struct and an enum,
// exported for other crates via `arael::export_models!()` at the bottom.
// Everything importable is pub with pub fields; `Hidden` deliberately is
// not, and rides the bundle as a tombstone.

use arael::model::{Component, CrossBlock, Param, SelfBlock};
use arael::matrix::matrix3;
use arael::quatern::quatern;
use arael::refs::Ref;
use arael::utils::Float;
use arael::matrix::matrix;
use arael::vect::{vect, vect2, vect3};

#[arael::model]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Fixed,
    Free,
}

/// Unit direction on S^2 with 2 DOF (the plane_slam_demo component).
#[arael::model]
#[arael(component)]
#[derive(Clone)]
pub struct Dir<T: Float> {
    /// Rotation taking (1, 0, 0) into `unit`. Solver-internal chart.
    pub ref_q: quatern<T>,
    #[arael(compute = self.ref_q.rotation_matrix())]
    pub rot: matrix3<T>,
    pub d: Param<vect2<T>>,
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    pub unit: vect3<T>,
    #[arael(deriv = unit, by = d)]
    pub unit_d: [vect3<T>; 2],
}

impl<T: Float> Dir<T> {
    fn ex() -> vect3<T> {
        vect3::new(T::one(), T::zero(), T::zero())
    }
    pub fn new(direction: vect3<T>) -> Dir<T> {
        let mut u = Dir {
            ref_q: quatern::identity(),
            rot: matrix3::identity(),
            d: Param::new(vect2::new(T::zero(), T::zero())),
            unit: direction,
            unit_d: [vect3::new(T::zero(), T::zero(), T::zero()); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl<T: Float> Component for Dir<T> {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quatern::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2::new(T::zero(), T::zero());
    }
    fn update(&mut self) {
        let dq = quatern::from_rotation_vector_small(
            vect3::new(T::zero(), self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2::new(T::zero(), T::zero());
    }
    fn finish(&mut self) {
        let dq = quatern::from_rotation_vector_small(
            vect3::new(T::zero(), self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

/// A beacon: 2D position plus a unit direction, each pulled to a prior.
#[arael::model]
#[arael(constraint(hb, {
    [(beacon.pos.x - beacon.prior.x) * 0.5,
     (beacon.pos.y - beacon.prior.y) * 0.5,
     beacon.dir.unit.x - beacon.target.x,
     beacon.dir.unit.y - beacon.target.y,
     beacon.dir.unit.z - beacon.target.z]
}))]
#[derive(Clone)]
pub struct Beacon<T: Float> {
    pub pos: Param<vect2<T>>,
    pub dir: Dir<T>,
    pub prior: vect2<T>,
    pub target: vect3<T>,
    pub kind: Kind,
    pub hb: SelfBlock<Beacon<T>, T>,
}

/// A spring between two beacons; the importing root must name its beacon
/// collection `beacons` (the resolution path is part of this contract).
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(sqrt(dx * dx + dy * dy) - spring.rest) * spring.w]
}, parent = spring))]
#[derive(Clone)]
pub struct Spring<T: Float> {
    #[arael(ref = root.beacons)]
    pub a: Ref<Beacon<T>>,
    #[arael(ref = root.beacons)]
    pub b: Ref<Beacon<T>>,
    pub rest: T,
    pub w: T,
    pub hb: CrossBlock<Beacon<T>, Beacon<T>, T>,
}

/// N-dof calibration entity: a const-generic `vect` parameter and a
/// `matrix` data field ride the export bundle. The prior rows pin every
/// component (the projected rows alone would leave the solve
/// underdetermined), so the optimum is exactly `t`.
#[arael::model]
#[arael(constraint(hb, {
    let d = cal.v - cal.t;
    let p = cal.h * d;
    [d[0] * cal.wp, d[1] * cal.wp, d[2] * cal.wp, d[3] * cal.wp,
     p[0] * cal.w, p[1] * cal.w]
}))]
#[derive(Clone)]
pub struct Cal<T: Float> {
    pub v: Param<vect<T, 4>>,
    pub t: vect<T, 4>,
    pub h: matrix<T, 2, 4>,
    pub wp: T,
    pub w: T,
    pub hb: SelfBlock<Cal<T>, T>,
}

/// Param-less record: a data-ref target riding the export bundle.
#[arael::model]
#[derive(Clone)]
pub struct Mark<T: Float> {
    pub anchor: T,
    pub w: T,
}

/// Pub struct with a private field: excluded from the bundle, imported
/// crates that reach for it get a tombstone error naming the field.
#[arael::model]
pub struct Hidden {
    secret: f64,
}

arael::export_models!();
