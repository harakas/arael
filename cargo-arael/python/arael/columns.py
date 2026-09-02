# arael Python column transport, shared by every generated module: how
# a whole collection column crosses the FFI in one call (set_<field>,
# get_<field>, push_many), and the sequence flattening push(**fields)
# applies to math-valued keywords. numpy is never imported at load;
# an array a caller passes is read through the buffer protocol, and
# get_<field> returns numpy only when it is importable.

import ctypes

# column code -> ctypes element, accepted buffer formats, item size, name
_CT = {"d": ctypes.c_double, "f": ctypes.c_float, "I": ctypes.c_uint32,
       "i": ctypes.c_int32, "B": ctypes.c_uint8}
_ACCEPT = {"d": ("d",), "f": ("f",), "I": ("I", "L", "Q"),
           "i": ("i", "l", "q"), "B": ("B", "b", "?")}
_SIZE = {"d": 8, "f": 4, "I": 4, "i": 4, "B": 1}
_NAME = {"d": "float64", "f": "float32", "I": "uint32", "i": "int32",
         "B": "bool"}

_np = False  # False: not tried yet; None: not importable


def numpy():
    """The numpy module when importable, else None (tried once)."""
    global _np
    if _np is False:
        try:
            import numpy
            _np = numpy
        except ImportError:
            _np = None
    return _np


def flat(v, n):
    """`v` as a tuple of n scalars: a flat sequence of n (a tuple, a
    numpy row, a math value), or rows that flatten to n (a matrix)."""
    t = tuple(v)
    if len(t) != n:
        t = tuple(x for row in t for x in row)
        if len(t) != n:
            raise TypeError("expected %d values, got %r" % (n, v))
    return t


def _scalar(v, code):
    if code == "B":
        return 1 if v else 0
    if code in ("I", "i"):
        return v.raw if hasattr(v, "raw") else int(v)
    return float(v)


def _address(v, mv):
    """The buffer's address, or None when it has to be copied first."""
    c = getattr(v, "ctypes", None)  # numpy: strided views included
    if c is not None:
        return c.data
    if mv.c_contiguous:
        try:
            return ctypes.addressof(ctypes.c_char.from_buffer(mv))
        except TypeError:  # read-only
            pass
    return None


def column_in(v, code, ncomp, n, name):
    """`v` presented as an n-element column of ncomp scalars each:
    (pointer, stride in bytes, object to keep alive). One scalar or one
    math value broadcasts (stride 0); a buffer of the right format is
    read in place; any other sequence is packed into a ctypes array."""
    ct = _CT[code]
    size = _SIZE[code]
    if not hasattr(v, "__len__"):
        if ncomp != 1:
            raise TypeError("%s: expected %d values per element, got %r"
                            % (name, ncomp, v))
        buf = ct(_scalar(v, code))
        return ctypes.addressof(buf), 0, buf
    try:
        mv = memoryview(v)
    except TypeError:
        mv = None
    if mv is not None:
        fmt = mv.format.lstrip("@=<>!")
        if fmt not in _ACCEPT[code] or mv.itemsize != size:
            raise TypeError("%s: expected %s values, got buffer format %r"
                            % (name, _NAME[code], mv.format))
        shape = tuple(mv.shape)
        if ncomp == 1 and mv.ndim == 1 and shape[0] == n:
            stride = mv.strides[0]
        elif (ncomp > 1 and mv.ndim == 2 and shape == (n, ncomp)
              and mv.strides[1] == size):
            stride = mv.strides[0]
        elif ncomp > 1 and mv.ndim == 1 and shape[0] == ncomp:
            stride = 0
        else:
            want = "(%d,)" % n if ncomp == 1 else "(%d, %d)" % (n, ncomp)
            raise ValueError("%s: expected shape %s, got %r"
                             % (name, want, shape))
        addr = _address(v, mv)
        if addr is not None:
            return addr, stride, v
        buf = ctypes.create_string_buffer(mv.tobytes(), mv.nbytes)
        return ctypes.addressof(buf), ncomp * size, buf
    if ncomp > 1 and len(v) == ncomp and not hasattr(v[0], "__len__"):
        buf = (ct * ncomp)(*[float(x) for x in v])
        return ctypes.addressof(buf), 0, buf
    if len(v) != n:
        raise ValueError("%s: expected %d values, got %d" % (name, n, len(v)))
    if ncomp == 1:
        buf = (ct * n)(*[_scalar(x, code) for x in v])
    else:
        buf = (ct * (n * ncomp))(*[float(x) for row in v
                                   for x in flat(row, ncomp)])
    return ctypes.addressof(buf), ncomp * size, buf


def column_out(code, ncomp, n):
    """A fresh buffer for an n-element column: (buffer, pointer,
    stride in bytes)."""
    buf = (_CT[code] * (n * ncomp))()
    return buf, ctypes.addressof(buf), ncomp * _SIZE[code]


def column_finish(buf, code, ncomp, n):
    """The filled buffer as the value get_<field> returns: a numpy
    array over it when numpy is importable (shape (n,) or (n, ncomp)),
    else the flat ctypes array itself."""
    np = numpy()
    if np is None:
        return buf
    dt = {"d": np.float64, "f": np.float32, "I": np.uint32, "i": np.int32,
          "B": np.bool_}[code]
    a = np.frombuffer(buf, dtype=dt)
    return a.reshape(n, ncomp) if ncomp > 1 else a


def count(n, pairs):
    """push_many's element count: `n` as given, else the length of the
    first keyword that is a sequence."""
    if n is not None:
        return int(n)
    for name, v in pairs:
        if v is not None and hasattr(v, "__len__") and not hasattr(v, "raw"):
            return len(v)
    raise TypeError("push_many: n is needed when no keyword is a sequence")
