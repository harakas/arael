// Runtime behavior of the const-generic `vect<T, N>` / `matrix<T, R, C>`
// types: operator identities, mixed ops with the fixed 2/3 types,
// conversions, serde, and the ParamType integration.

use arael::matrix::{matrix, matrix2f, matrix3d, matrixd};
use arael::model::{Param, ParamType};
use arael::vect::{vect, vect2f, vect3d, vectd, vectf};

fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-12 }

#[test]
fn vect_ops() {
    let a = vectd::<4>::new([1.0, 2.0, 3.0, 4.0]);
    let b = vectd::<4>::new([0.5, -1.0, 2.0, 0.0]);
    assert_eq!((a + b).e, [1.5, 1.0, 5.0, 4.0]);
    assert_eq!((a - b).e, [0.5, 3.0, 1.0, 4.0]);
    assert_eq!((-a).e, [-1.0, -2.0, -3.0, -4.0]);
    assert_eq!((a * 2.0).e, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!((2.0 * a).e, (a * 2.0).e);
    assert_eq!((a / 2.0).e, [0.5, 1.0, 1.5, 2.0]);
    assert!(close(a * b, 0.5 - 2.0 + 6.0));
    assert!(close(a.norm_squared(), 30.0));
    assert!(close(a.norm(), 30.0f64.sqrt()));
    assert_eq!(a[2], 3.0);
    let mut c = a;
    c[0] = 9.0;
    assert_eq!(c.e, [9.0, 2.0, 3.0, 4.0]);
    assert_eq!(vectd::<3>::zeros().e, [0.0; 3]);
    let f: vectf<4> = a.cast();
    assert_eq!(f.e, [1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn matrix_ops() {
    let a = matrixd::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = matrixd::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let ab = a * b;
    assert_eq!(ab.rows[0].e, [58.0, 64.0]);
    assert_eq!(ab.rows[1].e, [139.0, 154.0]);
    assert_eq!(a.transpose().rows[2].e, [3.0, 6.0]);
    let v = vectd::<3>::new([1.0, 0.0, -1.0]);
    assert_eq!((a * v).e, [-2.0, -2.0]);
    // (A * B) * x == A * (B * x)
    let x = vectd::<2>::new([0.3, -0.7]);
    let lhs = (a * b) * x;
    let rhs = a * (b * x);
    for i in 0..2 { assert!(close(lhs[i], rhs[i])); }
    let i3 = matrixd::<3, 3>::identity();
    assert_eq!((i3 * v).e, v.e);
    assert_eq!((a + a).rows[1].e, [8.0, 10.0, 12.0]);
    assert_eq!((a - a).rows[0].e, [0.0; 3]);
    assert_eq!((a * 2.0).rows[0].e, [2.0, 4.0, 6.0]);
    assert_eq!((2.0 * a).rows[0].e, (a * 2.0).rows[0].e);
    assert_eq!(a[1][2], 6.0);
}

#[test]
fn mixed_fixed_generic_ops() {
    // matrix<R, 2> * vect2 -> vect<R>
    let m = matrix::<f32, 3, 2>::from_array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    let v2 = vect2f::new(2.0, 3.0);
    let r = m * v2;
    assert_eq!(r.e, [2.0, 3.0, 5.0]);
    // matrix2 * vect<2> -> vect2
    let m2 = matrix2f::rotation(std::f32::consts::FRAC_PI_2);
    let gv = vectf::<2>::new([1.0, 0.0]);
    let rv = m2 * gv;
    assert!((rv.x - 0.0).abs() < 1e-6 && (rv.y - 1.0).abs() < 1e-6);
    // matrix3 * vect<3> -> vect3
    let m3 = matrix3d::identity();
    let g3 = vectd::<3>::new([1.0, 2.0, 3.0]);
    let r3 = m3 * g3;
    assert_eq!((r3.x, r3.y, r3.z), (1.0, 2.0, 3.0));
    // matrix<R, 3> * vect3 -> vect<R>
    let mr3 = matrixd::<2, 3>::from_array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let v3 = vect3d::new(4.0, 5.0, 6.0);
    assert_eq!((mr3 * v3).e, [4.0, 6.0]);
    // mixed matrix-matrix, both sides
    let gm = matrixd::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let f3 = matrix3d::identity();
    assert_eq!((gm * f3).rows[1].e, gm.rows[1].e);
    let f2 = arael::matrix::matrix2d::identity();
    assert_eq!((f2 * gm).rows[0].e, gm.rows[0].e);
    // mixed dot
    assert!(close(vectd::<3>::new([1.0, 2.0, 3.0]) * v3, 4.0 + 10.0 + 18.0));
    assert!(close(v3 * vectd::<3>::new([1.0, 2.0, 3.0]), 4.0 + 10.0 + 18.0));
}

#[test]
fn conversions() {
    let v2 = vect2f::new(1.0, 2.0);
    let g: vect<f32, 2> = v2.into();
    let back: vect2f = g.into();
    assert_eq!((back.x, back.y), (1.0, 2.0));
    let v3 = vect3d::new(1.0, 2.0, 3.0);
    let g3: vect<f64, 3> = v3.into();
    assert_eq!(g3.e, [1.0, 2.0, 3.0]);
    let m3 = matrix3d::identity();
    let gm: matrix<f64, 3, 3> = m3.into();
    let back3: matrix3d = gm.into();
    assert_eq!(back3.rows[1].y, 1.0);
    // nalgebra round trips
    let nv: nalgebra::SVector<f64, 3> = g3.into();
    let g3b: vect<f64, 3> = nv.into();
    assert_eq!(g3b.e, g3.e);
    let nm: nalgebra::SMatrix<f64, 3, 3> = gm.into();
    let gmb: matrix<f64, 3, 3> = nm.into();
    assert_eq!(gmb, gm);
}

#[test]
fn serde_roundtrip() {
    let v = vectd::<4>::new([1.0, -2.5, 3.25, 0.0]);
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, "[1.0,-2.5,3.25,0.0]");
    let back: vectd<4> = serde_json::from_str(&s).unwrap();
    assert_eq!(back, v);
    let m = matrixd::<2, 2>::from_array([[1.0, 2.0], [3.0, 4.0]]);
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(s, "[[1.0,2.0],[3.0,4.0]]");
    let back: matrixd<2, 2> = serde_json::from_str(&s).unwrap();
    assert_eq!(back, m);
}

#[test]
fn symmetric_eigen_reconstructs() {
    // A = R D R^T for a symmetric 4x4; eigen must reconstruct it.
    let a = matrixd::<4, 4>::from_array([
        [4.0, 1.0, 0.5, 0.0],
        [1.0, 3.0, 0.2, 0.1],
        [0.5, 0.2, 2.0, 0.3],
        [0.0, 0.1, 0.3, 1.0],
    ]);
    let (r, d) = a.symmetric_eigen();
    // ascending eigenvalues
    for k in 1..4 { assert!(d[k] >= d[k - 1]); }
    // reconstruct: sum_k d_k * (col_k col_k^T)
    let rt = r.transpose();
    let mut diag = matrixd::<4, 4>::zeros();
    for k in 0..4 { diag[k][k] = d[k]; }
    let rec = r * diag * rt;
    for i in 0..4 {
        for j in 0..4 {
            assert!((rec[i][j] - a[i][j]).abs() < 1e-9, "rec[{i}][{j}]");
        }
    }
}

#[test]
fn param_type_integration() {
    assert_eq!(<vectd<5> as ParamType>::SIZE, 5);
    let v = vectd::<5>::new([1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut buf = [0.0f32; 5];
    v.write_to(&mut buf);
    assert_eq!(buf, [1.0, 2.0, 3.0, 4.0, 5.0]);
    let back = <vectd<5> as ParamType>::read_from(&buf);
    assert_eq!(back, v);
    // Param over vect<N>: indexed component names.
    let mut names = Vec::new();
    <Param<vectd<3>> as arael::model::Model>::param_symbols("p", &mut names);
    assert_eq!(names, ["p[0]", "p[1]", "p[2]"]);
}
