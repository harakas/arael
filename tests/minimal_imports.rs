// The minimal import surface to define a model and solve it -- pinned as
// a test so API changes that grow the required imports get noticed.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [m.x - 3.0] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

#[test]
fn two_use_lines_suffice() {
    let mut m = M { x: Param::new(0.0), hb: SelfBlock::new() };
    let result = m.solve_sparse(&LmConfig::default()).unwrap();
    assert!(result.end_cost < 1e-12);
    assert!((m.x.value - 3.0).abs() < 1e-6);
}

// The same program through the prelude: one glob import covers it.
mod via_prelude {
    use arael::prelude::*;

    #[arael::model]
    #[arael(root)]
    #[arael(constraint(hb, { [p.x - p.t.x, p.y - p.t.y] }))]
    struct P {
        x: Param<f64>,
        y: Param<f64>,
        t: vect2d,
        hb: SelfBlock<P>,
    }

    #[test]
    fn one_glob_import_suffices() {
        let mut p = P { x: Param::new(0.0), y: Param::new(0.0),
                        t: vect2d::new(1.0, 2.0), hb: SelfBlock::new() };
        let result = p.solve_sparse(&LmConfig::default()).unwrap();
        assert!(result.end_cost < 1e-12);
        assert!((p.x.value - 1.0).abs() < 1e-6 && (p.y.value - 2.0).abs() < 1e-6);
    }
}

// The compound parameter types through the prelude too. The macro
// classifies a field by the last segment of its type path, so the name
// has to be in scope under exactly the spelling used here.
mod compound_via_prelude {
    use arael::prelude::*;

    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

    #[arael::model]
    #[arael(root)]
    // The transform is pinned by a measured translation and by where it
    // sends two axes; the direction by a measured direction. Every block is
    // over-determined (3 residuals against 3, 3 and 2 free parameters) and
    // exactly satisfiable, since the two measured axes are orthonormal and
    // the measured direction is unit length.
    #[arael(constraint(hb, {
        let dt = x.pose.translation - x.measured_translation;
        let fwd = x.pose.rotation_matrix * vect3sym::from_components(1.0, 0.0, 0.0)
            - x.measured_forward;
        let up = x.pose.rotation_matrix * vect3sym::from_components(0.0, 0.0, 1.0)
            - x.measured_up;
        let dir = x.dir.unit - x.measured_direction;
        [dt.x, dt.y, dt.z,
         fwd.x, fwd.y, fwd.z,
         up.x, up.y, up.z,
         dir.x, dir.y, dir.z]
    }))]
    struct X {
        pose: TransformParam,
        dir: UnitVecParam,
        measured_translation: vect3d,
        measured_forward: vect3d,
        measured_up: vect3d,
        measured_direction: vect3d,
        hb: SelfBlock<X>,
    }

    #[test]
    fn compound_params_reach_the_prelude() {
        let mut x = X {
            pose: TransformParam::new(vect3d::new(0.0, 0.0, 0.0), quaternd::identity()),
            dir: UnitVecParam::new(vect3d::new(1.0, 0.0, 0.0)),
            // A 45-degree yaw, and a direction 45 degrees off x toward z.
            measured_translation: vect3d::new(1.0, 2.0, 3.0),
            measured_forward: vect3d::new(SQRT_HALF, SQRT_HALF, 0.0),
            measured_up: vect3d::new(0.0, 0.0, 1.0),
            measured_direction: vect3d::new(SQRT_HALF, 0.0, SQRT_HALF),
            hb: SelfBlock::new(),
        };
        let result = x.solve_sparse(&LmConfig::default()).unwrap();
        assert!(result.end_cost < 1e-12, "end cost {}", result.end_cost);
        assert!((x.pose.translation - x.measured_translation).norm() < 1e-6);
        assert!((x.pose.rotation.rotation_matrix() * vect3d::new(1.0, 0.0, 0.0)
            - x.measured_forward).norm() < 1e-6);
        assert!((x.dir.unit - x.measured_direction).norm() < 1e-6);
        // The direction is a unit vector whatever the solve did to it.
        assert!((x.dir.unit.norm() - 1.0).abs() < 1e-12);
    }
}
