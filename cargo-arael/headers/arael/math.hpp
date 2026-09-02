// arael C++ math umbrella: vect2/3, matrix2/3, quatern and the rigid /
// scaled transforms with the same fields, operations, and conventions
// as arael's Rust types. `*` between vectors is DOT, `%` is CROSS;
// euler angles are x=roll, y=pitch, z=yaw with R = R(z)*R(y)*R(x);
// quaternions store the scalar part first.
#pragma once

#include "vect.hpp"
#include "matrix.hpp"
#include "quatern.hpp"
#include "transform.hpp"
