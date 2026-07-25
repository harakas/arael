# arael Python math: vect2/3, matrix2/3, quatern at f32 and f64.
# Mirrors arael's Rust types: the classes ARE the repr(C) structs
# (ctypes.Structure), so values cross the FFI without conversion.
# Same conventions: row-major matrices, euler x=roll y=pitch z=yaw
# with R = R(z)R(y)R(x), quaternion scalar part first. Constructors
# accept any length-matching sequence. Arithmetic runs in Python
# floats (f64) and rounds through the storage type on construction.

import ctypes
import math as _pm


def _seq(v, n):
    t = tuple(v)
    if len(t) != n:
        raise TypeError("expected a %d-sequence, got %r" % (n, v))
    return t


def _make_vect2(name, ct):
    class vect2(ctypes.Structure):
        _fields_ = [("x", ct), ("y", ct)]

        def __init__(self, x=0.0, y=0.0):
            if hasattr(x, "__len__") or hasattr(x, "__iter__"):
                x, y = _seq(x, 2)
            super().__init__(float(x), float(y))

        def __add__(self, o): return type(self)(self.x + o.x, self.y + o.y)
        def __sub__(self, o): return type(self)(self.x - o.x, self.y - o.y)
        def __neg__(self): return type(self)(-self.x, -self.y)

        def __mul__(self, o):
            if isinstance(o, vect2):
                return self.x * o.x + self.y * o.y  # dot
            if hasattr(o, "col"):  # row vector times matrix
                return type(self)(self * o.col(0), self * o.col(1))
            return type(self)(self.x * o, self.y * o)

        def __rmul__(self, s): return type(self)(self.x * s, self.y * s)
        def square(self): return self.x * self.x + self.y * self.y
        def norm(self): return _pm.sqrt(self.square())
        def unit(self): return self * (1.0 / self.norm())
        def across(self): return type(self)(-self.y, self.x)
        def cross(self, o): return self.x * o.y - self.y * o.x
        def deg2rad(self): return self * (_pm.pi / 180.0)
        def rad2deg(self): return self * (180.0 / _pm.pi)

        def __len__(self): return 2
        def __iter__(self): return iter((self.x, self.y))
        def __getitem__(self, i): return (self.x, self.y)[i]
        def __repr__(self): return "%s(%g, %g)" % (name, self.x, self.y)

    vect2.__name__ = vect2.__qualname__ = name
    return vect2


def _make_vect3(name, ct):
    class vect3(ctypes.Structure):
        _fields_ = [("x", ct), ("y", ct), ("z", ct)]

        def __init__(self, x=0.0, y=0.0, z=0.0):
            if hasattr(x, "__len__") or hasattr(x, "__iter__"):
                x, y, z = _seq(x, 3)
            super().__init__(float(x), float(y), float(z))

        def __add__(self, o):
            return type(self)(self.x + o.x, self.y + o.y, self.z + o.z)

        def __sub__(self, o):
            return type(self)(self.x - o.x, self.y - o.y, self.z - o.z)

        def __neg__(self): return type(self)(-self.x, -self.y, -self.z)

        def __mul__(self, o):
            if isinstance(o, vect3):
                return self.x * o.x + self.y * o.y + self.z * o.z  # dot
            if hasattr(o, "col"):  # row vector times matrix
                return type(self)(self * o.col(0), self * o.col(1),
                                  self * o.col(2))
            return type(self)(self.x * o, self.y * o, self.z * o)

        def __rmul__(self, s):
            return type(self)(self.x * s, self.y * s, self.z * s)

        def __mod__(self, o):  # cross
            return type(self)(
                self.y * o.z - self.z * o.y,
                self.z * o.x - self.x * o.z,
                self.x * o.y - self.y * o.x)

        def square(self):
            return self.x * self.x + self.y * self.y + self.z * self.z

        def norm(self): return _pm.sqrt(self.square())
        def unit(self): return self * (1.0 / self.norm())

        def across(self):
            # A unit vector orthogonal to self, as in Rust/C++.
            if abs(self.y) < abs(self.x):
                return type(self)(-self.z, 0.0, self.x).unit()
            return type(self)(0.0, self.z, -self.y).unit()

        def deg2rad(self): return self * (_pm.pi / 180.0)
        def rad2deg(self): return self * (180.0 / _pm.pi)

        def rotation_matrix(self):
            m3 = matrix3f if type(self) is vect3f else matrix3d
            return m3.rotation_from_euler_angles(self)

        def __len__(self): return 3
        def __iter__(self): return iter((self.x, self.y, self.z))
        def __getitem__(self, i): return (self.x, self.y, self.z)[i]

        def __repr__(self):
            return "%s(%g, %g, %g)" % (name, self.x, self.y, self.z)

    vect3.__name__ = vect3.__qualname__ = name
    return vect3


vect2f = _make_vect2("vect2f", ctypes.c_float)
vect2d = _make_vect2("vect2d", ctypes.c_double)
vect3f = _make_vect3("vect3f", ctypes.c_float)
vect3d = _make_vect3("vect3d", ctypes.c_double)

vect2f.cast = lambda self: vect2d(self.x, self.y)
vect2d.cast = lambda self: vect2f(self.x, self.y)
vect3f.cast = lambda self: vect3d(self.x, self.y, self.z)
vect3d.cast = lambda self: vect3f(self.x, self.y, self.z)


def _make_matrix2(name, vec):
    class matrix2(ctypes.Structure):
        _fields_ = [("rows", vec * 2)]

        def __init__(self, rows=None):
            super().__init__()
            if rows is not None:
                r0, r1 = _seq(rows, 2)
                self.rows[0] = r0 if isinstance(r0, vec) else vec(r0)
                self.rows[1] = r1 if isinstance(r1, vec) else vec(r1)

        @classmethod
        def from_rows(cls, r0, r1): return cls((r0, r1))

        @classmethod
        def from_cols(cls, c0, c1):
            return cls(((c0[0], c1[0]), (c0[1], c1[1])))

        @classmethod
        def from_elements(cls, a00, a01, a10, a11):
            return cls(((a00, a01), (a10, a11)))

        @classmethod
        def zero_matrix(cls): return cls.from_elements(0.0, 0.0, 0.0, 0.0)

        @classmethod
        def identity(cls): return cls.from_elements(1.0, 0.0, 0.0, 1.0)

        @classmethod
        def rotation(cls, angle):
            return cls.rotation_from_sincos(_pm.sin(angle), _pm.cos(angle))

        @classmethod
        def rotation_from_sincos(cls, s, c):
            return cls.from_elements(c, -s, s, c)

        def row(self, i): return vec(self.rows[i])
        def col(self, i): return vec((self.rows[0][i], self.rows[1][i]))
        def transpose(self): return type(self).from_cols(self.rows[0], self.rows[1])

        def det(self):
            return self.rows[0].x * self.rows[1].y - self.rows[0].y * self.rows[1].x

        def get_rotation_angle(self):
            return _pm.atan2(self.rows[1][0], self.rows[0][0])

        def __add__(self, m):
            return type(self)((self.row(0) + m.row(0), self.row(1) + m.row(1)))

        def __sub__(self, m):
            return type(self)((self.row(0) - m.row(0), self.row(1) - m.row(1)))

        def __neg__(self): return type(self)((-self.row(0), -self.row(1)))

        def __mul__(self, o):
            if isinstance(o, vec):
                return vec(self.row(0) * o, self.row(1) * o)
            if isinstance(o, matrix2):
                return type(self).from_rows(
                    vec(self.row(0) * o.col(0), self.row(0) * o.col(1)),
                    vec(self.row(1) * o.col(0), self.row(1) * o.col(1)))
            return type(self)((self.row(0) * o, self.row(1) * o))

        def __rmul__(self, s): return self * s
        def __getitem__(self, i): return self.row(i)

        def symmetric_eigen(self):
            # (R, d), eigenvector columns, eigenvalues ascending --
            # the C++ header's closed form.
            a = float(self.rows[0][0]); b = float(self.rows[0][1])
            c = float(self.rows[1][1])
            half = (a + c) * 0.5
            disc = _pm.sqrt((a - c) * (a - c) * 0.25 + b * b)
            l0 = half - disc; l1 = half + disc
            if abs(l1 - a) >= abs(l1 - c):
                vx, vy = b, l1 - a
            else:
                vx, vy = l1 - c, b
            n = _pm.sqrt(vx * vx + vy * vy)
            if n > 0.0:
                vx /= n; vy /= n
            else:
                vx, vy = 1.0, 0.0
            r = type(self).from_cols((-vy, vx), (vx, vy))
            v2 = vect2f if vec is vect3f else vect2d
            return r, v2(l0, l1)

        def __repr__(self):
            return "%s(%r, %r)" % (name, self.row(0), self.row(1))

    matrix2.__name__ = matrix2.__qualname__ = name
    return matrix2


def _make_matrix3(name, vec):
    class matrix3(ctypes.Structure):
        _fields_ = [("rows", vec * 3)]

        def __init__(self, rows=None):
            super().__init__()
            if rows is not None:
                r0, r1, r2 = _seq(rows, 3)
                self.rows[0] = r0 if isinstance(r0, vec) else vec(r0)
                self.rows[1] = r1 if isinstance(r1, vec) else vec(r1)
                self.rows[2] = r2 if isinstance(r2, vec) else vec(r2)

        @classmethod
        def from_rows(cls, r0, r1, r2): return cls((r0, r1, r2))

        @classmethod
        def from_cols(cls, c0, c1, c2):
            return cls(((c0[0], c1[0], c2[0]),
                        (c0[1], c1[1], c2[1]),
                        (c0[2], c1[2], c2[2])))

        @classmethod
        def from_elements(cls, a00, a01, a02, a10, a11, a12, a20, a21, a22):
            return cls(((a00, a01, a02), (a10, a11, a12), (a20, a21, a22)))

        @classmethod
        def zero_matrix(cls):
            return cls.from_elements(0, 0, 0, 0, 0, 0, 0, 0, 0)

        @classmethod
        def identity(cls):
            return cls.from_elements(1, 0, 0, 0, 1, 0, 0, 0, 1)

        @classmethod
        def rotation_from_euler_angles(cls, ea):
            s = (_pm.sin(ea[0]), _pm.sin(ea[1]), _pm.sin(ea[2]))
            c = (_pm.cos(ea[0]), _pm.cos(ea[1]), _pm.cos(ea[2]))
            return cls.rotation_from_euler_angles_sincos(s, c)

        @classmethod
        def rotation_from_euler_angles_sincos(cls, s, c):
            return cls.from_elements(
                c[2] * c[1], c[2] * s[1] * s[0] - s[2] * c[0],
                c[2] * s[1] * c[0] + s[2] * s[0],
                s[2] * c[1], s[2] * s[1] * s[0] + c[2] * c[0],
                s[2] * s[1] * c[0] - c[2] * s[0],
                -s[1], c[1] * s[0], c[1] * c[0])

        @classmethod
        def rotation_from_axis_angle(cls, axis, phi):
            return cls.rotation_from_axis_angle_sincos(
                axis, _pm.sin(phi), _pm.cos(phi))

        @classmethod
        def rotation_from_axis_angle_sincos(cls, a, sp, cp):
            k = 1.0 - cp
            return cls.from_elements(
                cp + a[0] * a[0] * k, a[0] * a[1] * k - a[2] * sp,
                a[0] * a[2] * k + a[1] * sp,
                a[1] * a[0] * k + a[2] * sp, cp + a[1] * a[1] * k,
                a[1] * a[2] * k - a[0] * sp,
                a[2] * a[0] * k - a[1] * sp, a[2] * a[1] * k + a[0] * sp,
                cp + a[2] * a[2] * k)

        @classmethod
        def from_rotation_vector_small(cls, w):
            # The normalize(1, v/2) retraction, sqrt-free.
            x = w[0] * 0.5; y = w[1] * 0.5; z = w[2] * 0.5
            x2 = x * x; y2 = y * y; z2 = z * z
            s = 2.0 / (1.0 + x2 + y2 + z2)
            return cls.from_elements(
                1.0 - s * (y2 + z2), s * (x * y - z), s * (x * z + y),
                s * (x * y + z), 1.0 - s * (x2 + z2), s * (y * z - x),
                s * (x * z - y), s * (y * z + x), 1.0 - s * (x2 + y2))

        def get_rotation_vector_small(self):
            return vec((self.rows[2][1] - self.rows[1][2]) * 0.5,
                       (self.rows[0][2] - self.rows[2][0]) * 0.5,
                       (self.rows[1][0] - self.rows[0][1]) * 0.5)

        def get_euler_angles(self):
            m20 = max(-1.0, min(1.0, float(self.rows[2][0])))
            y = _pm.asin(-m20)
            cp2 = (self.rows[2][1] * self.rows[2][1]
                   + self.rows[2][2] * self.rows[2][2])
            eps = 1.1920929e-07 if vec is vect3f else 2.220446049250313e-16
            if cp2 > eps:
                return vec(_pm.atan2(self.rows[2][1], self.rows[2][2]), y,
                           _pm.atan2(self.rows[1][0], self.rows[0][0]))
            return vec(0.0, y, _pm.atan2(-self.rows[0][1], self.rows[1][1]))

        def row(self, i): return vec(self.rows[i])

        def col(self, i):
            return vec((self.rows[0][i], self.rows[1][i], self.rows[2][i]))

        def transpose(self):
            return type(self).from_cols(self.rows[0], self.rows[1], self.rows[2])

        def det(self):
            r = self.rows
            return (r[0].x * (r[1].y * r[2].z - r[1].z * r[2].y)
                    - r[0].y * (r[1].x * r[2].z - r[1].z * r[2].x)
                    + r[0].z * (r[1].x * r[2].y - r[1].y * r[2].x))

        def is_finite(self):
            return all(_pm.isfinite(v) for r in self.rows for v in r)

        def __add__(self, m):
            return type(self)((self.row(0) + m.row(0), self.row(1) + m.row(1),
                               self.row(2) + m.row(2)))

        def __sub__(self, m):
            return type(self)((self.row(0) - m.row(0), self.row(1) - m.row(1),
                               self.row(2) - m.row(2)))

        def __neg__(self):
            return type(self)((-self.row(0), -self.row(1), -self.row(2)))

        def __mul__(self, o):
            if isinstance(o, vec):
                return vec(self.row(0) * o, self.row(1) * o, self.row(2) * o)
            if isinstance(o, matrix3):
                return type(self).from_rows(
                    *[vec(self.row(r) * o.col(0), self.row(r) * o.col(1),
                          self.row(r) * o.col(2)) for r in range(3)])
            return type(self)((self.row(0) * o, self.row(1) * o,
                               self.row(2) * o))

        def __rmul__(self, s): return self * s
        def __getitem__(self, i): return self.row(i)

        def symmetric_eigen(self):
            # (R, d), eigenvector columns, eigenvalues ascending --
            # cyclic Jacobi in double, the C++ header's algorithm.
            a = [[float(self.rows[r][c]) for c in range(3)] for r in range(3)]
            v = [[1.0 if r == c else 0.0 for c in range(3)] for r in range(3)]
            for _ in range(64):
                off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
                if not off > 1e-300:
                    break
                for p in range(2):
                    for q in range(p + 1, 3):
                        if a[p][q] == 0.0:
                            continue
                        theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                        t = ((1.0 if theta >= 0.0 else -1.0)
                             / (abs(theta) + _pm.sqrt(theta * theta + 1.0)))
                        c = 1.0 / _pm.sqrt(t * t + 1.0)
                        s = t * c
                        for k in range(3):
                            akp, akq = a[k][p], a[k][q]
                            a[k][p] = c * akp - s * akq
                            a[k][q] = s * akp + c * akq
                        for k in range(3):
                            apk, aqk = a[p][k], a[q][k]
                            a[p][k] = c * apk - s * aqk
                            a[q][k] = s * apk + c * aqk
                        for k in range(3):
                            vkp, vkq = v[k][p], v[k][q]
                            v[k][p] = c * vkp - s * vkq
                            v[k][q] = s * vkp + c * vkq
            idx = sorted(range(3), key=lambda i: a[i][i])
            d = vec(a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]])
            r = type(self).from_cols(
                *[(v[0][j], v[1][j], v[2][j]) for j in idx])
            return r, d

        def __repr__(self):
            return "%s(%r, %r, %r)" % (name, self.row(0), self.row(1),
                                       self.row(2))

    matrix3.__name__ = matrix3.__qualname__ = name
    return matrix3


matrix2f = _make_matrix2("matrix2f", vect2f)
matrix2d = _make_matrix2("matrix2d", vect2d)
matrix3f = _make_matrix3("matrix3f", vect3f)
matrix3d = _make_matrix3("matrix3d", vect3d)

matrix2f.cast = lambda self: matrix2d((self.row(0), self.row(1)))
matrix2d.cast = lambda self: matrix2f((self.row(0), self.row(1)))
matrix3f.cast = lambda self: matrix3d((self.row(0), self.row(1), self.row(2)))
matrix3d.cast = lambda self: matrix3f((self.row(0), self.row(1), self.row(2)))


def _clamp1(v):
    return max(-1.0, min(1.0, float(v)))


def _rad2rad(x):
    if -_pm.pi <= x <= _pm.pi:
        return x
    return x - 2.0 * _pm.pi * _pm.floor((x - 0.0) / (2.0 * _pm.pi) + 0.5)


def _make_quatern(name, vec, ct, eps):
    # Exact port of the C++ arael/quatern.hpp formulas (which mirror
    # src/quatern.rs): the parity test holds them to the same values.
    class quatern(ctypes.Structure):
        _fields_ = [("t", ct), ("v", vec)]

        def __init__(self, t=1.0, v=None):
            super().__init__()
            self.t = float(t)
            if isinstance(v, vec):
                self.v = v
            else:
                self.v = vec(v) if v is not None else vec()

        @classmethod
        def identity(cls): return cls(1.0, (0.0, 0.0, 0.0))

        def dot(self, q): return float(self.t * q.t + self.v * q.v)
        def norm(self): return _pm.sqrt(self.dot(self))

        def unit(self):
            k = 1.0 / self.norm()
            return type(self)(self.t * k, self.v * k)

        def conj(self): return type(self)(self.t, -self.v)

        def __mul__(self, o):
            if isinstance(o, quatern):
                return type(self)(
                    self.t * o.t - self.v * o.v,
                    o.v * self.t + self.v * o.t + (self.v % o.v))
            return type(self)(self.t * o, self.v * o)

        def __rmul__(self, s): return self * s
        def __add__(self, q): return type(self)(self.t + q.t, self.v + q.v)
        def __sub__(self, q): return type(self)(self.t - q.t, self.v - q.v)
        def __neg__(self): return type(self)(-self.t, -self.v)

        def rotate(self, p):
            return (self * type(self)(0.0, p) * self.conj()).v

        def rotation_matrix(self):
            m3 = matrix3f if vec is vect3f else matrix3d
            t, x, y, z = (float(self.t), float(self.v.x), float(self.v.y),
                          float(self.v.z))
            x2 = x * x; y2 = y * y; z2 = z * z
            return m3.from_rows(
                (1.0 - 2.0 * (y2 + z2), 2.0 * (x * y - z * t),
                 2.0 * (x * z + y * t)),
                (2.0 * (x * y + z * t), 1.0 - 2.0 * (x2 + z2),
                 2.0 * (y * z - x * t)),
                (2.0 * (x * z - y * t), 2.0 * (y * z + x * t),
                 1.0 - 2.0 * (x2 + y2)))

        def get_axis_angle(self):
            angle = _rad2rad(2.0 * _pm.acos(_clamp1(self.t)))
            s2 = 1.0 - float(self.t) * float(self.t)
            if s2 > eps * eps:
                return self.v * (1.0 / _pm.sqrt(s2)), angle
            return vec(1.0, 0.0, 0.0), 0.0

        def get_euler_angles(self):
            t, x, y, z = (float(self.t), float(self.v.x), float(self.v.y),
                          float(self.v.z))
            pitch = _pm.asin(_clamp1(2.0 * (t * y - z * x)))
            roll_num = 2.0 * (t * x + y * z)
            roll_den = 1.0 - 2.0 * (x * x + y * y)
            cp2 = roll_num * roll_num + roll_den * roll_den
            if cp2 > eps:
                return vec(_pm.atan2(roll_num, roll_den), pitch,
                           _pm.atan2(2.0 * (t * z + x * y),
                                     1.0 - 2.0 * (y * y + z * z)))
            return vec(0.0, pitch,
                       _pm.atan2(2.0 * (t * z - x * y),
                                 1.0 - 2.0 * (x * x + z * z)))

        @classmethod
        def from_euler_angles(cls, ea):
            ha = [float(a) * 0.5 for a in _seq(ea, 3)]
            shax, chax = _pm.sin(ha[0]), _pm.cos(ha[0])
            shay, chay = _pm.sin(ha[1]), _pm.cos(ha[1])
            shaz, chaz = _pm.sin(ha[2]), _pm.cos(ha[2])
            return cls(chax * chay * chaz + shax * shay * shaz,
                       (shax * chay * chaz - chax * shay * shaz,
                        chax * shay * chaz + shax * chay * shaz,
                        chax * chay * shaz - shax * shay * chaz))

        @classmethod
        def from_axis_angle(cls, normal, angle):
            half = 0.5 * float(angle)
            return cls(_pm.cos(half), vec(normal) * _pm.sin(half))

        @classmethod
        def from_rotation_vector(cls, w):
            v3 = vec(w)
            s = v3.square()
            theta = _pm.sqrt(s)
            half = 0.5 * theta
            if s >= 1e-8:
                qt = _pm.cos(half)
                scale = _pm.sin(half) / theta
            else:
                qt = 1.0 - s * 0.125 + s * s * (1.0 / 384.0)
                scale = 0.5 - s * (1.0 / 48.0)
            return cls(qt, v3 * scale)

        @classmethod
        def from_rotation_vector_small(cls, w):
            return cls(1.0, vec(w) * 0.5).unit()

        @classmethod
        def from_rotation_matrix(cls, m):
            # Shepperd's method: largest squared component first.
            tr = float(m[0].x + m[1].y + m[2].z)
            if tr > 0.0:
                s = _pm.sqrt(tr + 1.0) * 2.0
                q = cls(0.25 * s, ((m[2].y - m[1].z) / s,
                                   (m[0].z - m[2].x) / s,
                                   (m[1].x - m[0].y) / s))
            elif m[0].x > m[1].y and m[0].x > m[2].z:
                s = _pm.sqrt(1.0 + m[0].x - m[1].y - m[2].z) * 2.0
                q = cls((m[2].y - m[1].z) / s,
                        (0.25 * s, (m[0].y + m[1].x) / s,
                         (m[0].z + m[2].x) / s))
            elif m[1].y > m[2].z:
                s = _pm.sqrt(1.0 + m[1].y - m[0].x - m[2].z) * 2.0
                q = cls((m[0].z - m[2].x) / s,
                        ((m[0].y + m[1].x) / s, 0.25 * s,
                         (m[1].z + m[2].y) / s))
            else:
                s = _pm.sqrt(1.0 + m[2].z - m[0].x - m[1].y) * 2.0
                q = cls((m[1].x - m[0].y) / s,
                        ((m[0].z + m[2].x) / s, (m[1].z + m[2].y) / s,
                         0.25 * s))
            return q.unit()

        def pow(self, f):
            axis, angle = self.get_axis_angle()
            return type(self).from_axis_angle(axis, f * angle)

        def log(self):
            axis, angle = self.get_axis_angle()
            return type(self)(0.0, axis * angle)

        def exp(self):
            angle = self.v.norm()
            if angle < eps:
                return type(self).identity()
            return type(self).from_axis_angle(self.v * (1.0 / angle), angle)

        @classmethod
        def from_two_vectors(cls, from_v, to_v):
            f = vec(from_v); t = vec(to_v)
            mid = (f + t) * 0.5
            mid_len2 = mid * mid
            if mid_len2 < eps:
                return cls.from_axis_angle(f.across(), _pm.pi)
            mid = mid * (1.0 / _pm.sqrt(mid_len2))
            return cls(mid * t, mid % t)

        @classmethod
        def slerp(cls, from_q, to_q, f):
            return from_q * (from_q.conj() * to_q).pow(f)

        def __repr__(self):
            return "%s(%g, %r)" % (name, self.t, self.v)

    quatern.__name__ = quatern.__qualname__ = name
    return quatern


quaternf = _make_quatern("quaternf", vect3f, ctypes.c_float, 1.1920929e-07)
quaternd = _make_quatern("quaternd", vect3d, ctypes.c_double,
                         2.220446049250313e-16)
