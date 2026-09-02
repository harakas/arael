// arael C++ transforms: rigid and scaled transforms as plain values,
// mirroring arael's Rust `transform3` / `scaled_transform3`, and the
// live views a generated wrapper hands out for a TransformParam /
// ScaledTransformParam field. A view holds the element pointer and the
// field's accessors; its parts read and write through, and the
// transform algebra (`*`, inv(), compose ...) runs here on the current
// parts. A rigid transform acts on a point as R x + t, a scaled one as
// s (R x) + t.
#pragma once

#include "vect.hpp"
#include "matrix.hpp"
#include "quatern.hpp"

namespace arael {

template<class T> struct scaled_transform3;

/// A rigid transform: rotation matrix and translation. `inv()` and
/// rigid compositions return the same type; no scale anywhere.
template<class T>
struct transform3 {
    matrix3<T> rotation_matrix;
    vect3<T> translation;

    /// From a translation and a quaternion rotation.
    static transform3 from(vect3<T> t, quatern<T> q) { return {q.rotation_matrix(), t}; }
    static transform3 identity() { return {matrix3<T>::identity(), {T(0), T(0), T(0)}}; }

    /// The rotation as a quaternion.
    quatern<T> rotation() const { return quatern<T>::from_rotation_matrix(rotation_matrix); }

    /// The action on a point: R x + t.
    vect3<T> transform(vect3<T> x) const { return rotation_matrix * x + translation; }
    /// The action of the inverse: R^T (y - t).
    vect3<T> inverse_transform(vect3<T> y) const {
        return rotation_matrix.transpose() * (y - translation);
    }
    /// The rotation alone: R v.
    vect3<T> rotate(vect3<T> v) const { return rotation_matrix * v; }
    /// The inverse rotation alone: R^T v.
    vect3<T> inverse_rotate(vect3<T> v) const { return rotation_matrix.transpose() * v; }

    /// The inverse transform: (R^T, -R^T t).
    transform3 inv() const {
        matrix3<T> rt = rotation_matrix.transpose();
        return {rt, -(rt * translation)};
    }
    /// `*this * b`: b applied first, then this.
    transform3 compose(const transform3& b) const {
        return {rotation_matrix * b.rotation_matrix, rotation_matrix * b.translation + translation};
    }
    scaled_transform3<T> compose(const scaled_transform3<T>& b) const;

    vect3<T> operator*(vect3<T> x) const { return transform(x); }
    transform3 operator*(const transform3& b) const { return compose(b); }
    scaled_transform3<T> operator*(const scaled_transform3<T>& b) const;
};

/// A scaled transform: rotation matrix, translation and a uniform
/// scale. Compositions with a rigid transform, from either side, are
/// scaled; the inverse action divides once per call.
template<class T>
struct scaled_transform3 {
    matrix3<T> rotation_matrix;
    vect3<T> translation;
    T scale;

    static scaled_transform3 from(vect3<T> t, quatern<T> q, T s) { return {q.rotation_matrix(), t, s}; }
    static scaled_transform3 from(const transform3<T>& r) { return {r.rotation_matrix, r.translation, T(1)}; }
    static scaled_transform3 identity() { return from(transform3<T>::identity()); }

    quatern<T> rotation() const { return quatern<T>::from_rotation_matrix(rotation_matrix); }

    /// The action on a point: s (R x) + t.
    vect3<T> transform(vect3<T> x) const { return rotation_matrix * x * scale + translation; }
    /// The action of the inverse: R^T (y - t) / s, one division.
    vect3<T> inverse_transform(vect3<T> y) const {
        T k = T(1) / scale;
        return rotation_matrix.transpose() * (y - translation) * k;
    }
    /// The rotation alone: R v, never scaled.
    vect3<T> rotate(vect3<T> v) const { return rotation_matrix * v; }
    /// The inverse rotation alone: R^T v, never scaled.
    vect3<T> inverse_rotate(vect3<T> v) const { return rotation_matrix.transpose() * v; }

    /// The inverse transform: (R^T, -R^T t / s, 1 / s), one division.
    scaled_transform3 inv() const {
        matrix3<T> rt = rotation_matrix.transpose();
        T k = T(1) / scale;
        return {rt, -(rt * translation) * k, k};
    }
    /// `*this * b`: b applied first, then this.
    scaled_transform3 compose(const scaled_transform3& b) const {
        return {rotation_matrix * b.rotation_matrix,
                rotation_matrix * b.translation * scale + translation,
                scale * b.scale};
    }
    scaled_transform3 compose(const transform3<T>& b) const { return compose(from(b)); }

    vect3<T> operator*(vect3<T> x) const { return transform(x); }
    scaled_transform3 operator*(const scaled_transform3& b) const { return compose(b); }
    scaled_transform3 operator*(const transform3<T>& b) const { return compose(b); }
};

template<class T>
inline scaled_transform3<T> transform3<T>::compose(const scaled_transform3<T>& b) const {
    return scaled_transform3<T>::from(*this).compose(b);
}
template<class T>
inline scaled_transform3<T> transform3<T>::operator*(const scaled_transform3<T>& b) const {
    return compose(b);
}

using transform3d = transform3<double>;
using transform3f = transform3<float>;
using scaled_transform3d = scaled_transform3<double>;
using scaled_transform3f = scaled_transform3<float>;

/// A TransformParam field of a generated wrapper, live: the parts read
/// and write through the field's accessors, `rotation_matrix()` is
/// computed, and the transform algebra runs on the current parts. It
/// converts to a `transform3` (its snapshot), so it composes with values
/// and other views alike: `a.r2w().inv() * b.r2w()`.
template<class E, class T>
struct TransformParamView {
    E* h_;
    vect3<T> (*get_translation)(const E*);
    void (*set_translation_)(E*, vect3<T>);
    quatern<T> (*get_rotation)(const E*);
    void (*set_rotation_)(E*, quatern<T>);
    bool (*get_optimize_translation)(const E*);
    void (*set_optimize_translation_)(E*, bool);
    bool (*get_optimize_rotation)(const E*);
    void (*set_optimize_rotation_)(E*, bool);

    vect3<T> translation() const { return get_translation(h_); }
    void set_translation(vect3<T> v) const { set_translation_(h_, v); }
    quatern<T> rotation() const { return get_rotation(h_); }
    void set_rotation(quatern<T> q) const { set_rotation_(h_, q); }
    bool optimize_translation() const { return get_optimize_translation(h_); }
    void set_optimize_translation(bool v) const { set_optimize_translation_(h_, v); }
    bool optimize_rotation() const { return get_optimize_rotation(h_); }
    void set_optimize_rotation(bool v) const { set_optimize_rotation_(h_, v); }
    matrix3<T> rotation_matrix() const { return rotation().rotation_matrix(); }

    /// The current pose as a value.
    transform3<T> to_transform() const { return transform3<T>::from(translation(), rotation()); }
    operator transform3<T>() const { return to_transform(); }

    vect3<T> transform(vect3<T> x) const { return to_transform().transform(x); }
    vect3<T> inverse_transform(vect3<T> y) const { return to_transform().inverse_transform(y); }
    vect3<T> rotate(vect3<T> v) const { return to_transform().rotate(v); }
    vect3<T> inverse_rotate(vect3<T> v) const { return to_transform().inverse_rotate(v); }
    transform3<T> inv() const { return to_transform().inv(); }
    template<class R> auto operator*(const R& r) const { return to_transform() * r; }
};

/// A ScaledTransformParam field of a generated wrapper, live, with
/// `scale()` and `optimize_scale()` beside the rigid parts. Converts to
/// a `scaled_transform3`.
template<class E, class T>
struct ScaledTransformParamView {
    E* h_;
    vect3<T> (*get_translation)(const E*);
    void (*set_translation_)(E*, vect3<T>);
    quatern<T> (*get_rotation)(const E*);
    void (*set_rotation_)(E*, quatern<T>);
    T (*get_scale)(const E*);
    void (*set_scale_)(E*, T);
    bool (*get_optimize_translation)(const E*);
    void (*set_optimize_translation_)(E*, bool);
    bool (*get_optimize_rotation)(const E*);
    void (*set_optimize_rotation_)(E*, bool);
    bool (*get_optimize_scale)(const E*);
    void (*set_optimize_scale_)(E*, bool);

    vect3<T> translation() const { return get_translation(h_); }
    void set_translation(vect3<T> v) const { set_translation_(h_, v); }
    quatern<T> rotation() const { return get_rotation(h_); }
    void set_rotation(quatern<T> q) const { set_rotation_(h_, q); }
    T scale() const { return get_scale(h_); }
    void set_scale(T s) const { set_scale_(h_, s); }
    bool optimize_translation() const { return get_optimize_translation(h_); }
    void set_optimize_translation(bool v) const { set_optimize_translation_(h_, v); }
    bool optimize_rotation() const { return get_optimize_rotation(h_); }
    void set_optimize_rotation(bool v) const { set_optimize_rotation_(h_, v); }
    bool optimize_scale() const { return get_optimize_scale(h_); }
    void set_optimize_scale(bool v) const { set_optimize_scale_(h_, v); }
    matrix3<T> rotation_matrix() const { return rotation().rotation_matrix(); }

    /// The current pose as a value.
    scaled_transform3<T> to_scaled_transform() const {
        return scaled_transform3<T>::from(translation(), rotation(), scale());
    }
    operator scaled_transform3<T>() const { return to_scaled_transform(); }

    vect3<T> transform(vect3<T> x) const { return to_scaled_transform().transform(x); }
    vect3<T> inverse_transform(vect3<T> y) const { return to_scaled_transform().inverse_transform(y); }
    vect3<T> rotate(vect3<T> v) const { return to_scaled_transform().rotate(v); }
    vect3<T> inverse_rotate(vect3<T> v) const { return to_scaled_transform().inverse_rotate(v); }
    scaled_transform3<T> inv() const { return to_scaled_transform().inv(); }
    template<class R> auto operator*(const R& r) const { return to_scaled_transform() * r; }
};

}  // namespace arael
