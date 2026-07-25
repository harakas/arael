# arael Python solver surface, shared by every generated module:
# status/preset/covariance enums, the per-precision config, result and
# iteration layouts (ctypes mirrors of the C ABI -- field order
# matters), and AraelError. A generated module instantiates the
# layouts at its root's precision via lm_types() and wires the config
# constructor to its root's FFI.

import ctypes
import enum


class AraelError(Exception):
    """A failed solve (status < 0), a caught Rust panic, or a failed
    covariance query; carries the status code and the last_error
    text."""

    def __init__(self, status, message):
        super().__init__("%s (status %d)" % (message, status))
        self.status = int(status)
        self.message = message


class LmStatus(enum.IntEnum):
    """Why a solve stopped (matches the Rust LmStatus mapping)."""
    CONVERGED = 0
    COST_THRESHOLD = 1
    MAX_ITERATIONS = 2
    GRADIENT_TOLERANCE = 3
    PARAMETER_TOLERANCE = 4
    PREDICTED_REDUCTION = 5
    LAMBDA_CEILING = 6
    DRIVER_TERMINATED = 7
    OBSERVER_TERMINATED = 8
    TIME_LIMIT = 9
    RETRY_BUDGET_EXHAUSTED = 10
    ABORTED = 11
    SOLVER_FAILED = -1
    PANICKED = -2


class LmPreset(enum.IntEnum):
    DEFAULTS = 0
    CONSERVATIVE = 1
    WELL_CONDITIONED = 2


class CovMode(enum.IntEnum):
    PER_QUERY = 0
    ALL_MARGINALS = 1
    TRI_DIAGONAL = 2


class LmTiming(ctypes.Structure):
    """Per-phase wall-clock seconds plus call counts, gathered when
    config.gather_timing is set."""
    _fields_ = [
        ("total", ctypes.c_double),
        ("assembly", ctypes.c_double),
        ("first_assembly", ctypes.c_double),
        ("analysis", ctypes.c_double),
        ("linear_solve", ctypes.c_double),
        ("first_linear_solve", ctypes.c_double),
        ("cost_eval", ctypes.c_double),
        ("first_cost_eval", ctypes.c_double),
        ("advance", ctypes.c_double),
        ("first_advance", ctypes.c_double),
        ("assembly_count", ctypes.c_uint32),
        ("analysis_count", ctypes.c_uint32),
        ("linear_solve_count", ctypes.c_uint32),
        ("cost_eval_count", ctypes.c_uint32),
        ("advance_count", ctypes.c_uint32),
    ]


def _opt_property(field):
    """Property mapping a COpt struct field to a float-or-None."""

    def get(self):
        c = getattr(self, field)
        return float(c.v) if c.has else None

    def set(self, value):
        c = getattr(self, field)
        if value is None:
            c.has = False
            c.v = 0.0
        else:
            c.has = True
            c.v = float(value)

    return property(get, set)


def lm_types(fp):
    """The per-precision solver layouts as a dict: fp is
    ctypes.c_float or ctypes.c_double. Field order is the C ABI."""

    class COptF(ctypes.Structure):
        _fields_ = [("has", ctypes.c_bool), ("v", fp)]

    class COptSeconds(ctypes.Structure):
        _fields_ = [("has", ctypes.c_bool), ("v", ctypes.c_double)]

    class LmIter(ctypes.Structure):
        """One damped attempt, as the observer callback sees it.
        `params`/`params_len` point at the CURRENT parameter vector;
        valid only during the callback."""
        _fields_ = [
            ("iter", ctypes.c_uint32),
            ("inner", ctypes.c_uint32),
            ("accepted", ctypes.c_bool),
            ("factorization_failed", ctypes.c_bool),
            ("cost", fp),
            ("new_cost", fp),
            ("lambda_", fp),
            ("accepted_total", ctypes.c_uint32),
            ("params", ctypes.POINTER(fp)),
            ("params_len", ctypes.c_uint32),
        ]

        def param(self, i):
            return float(self.params[i])

        def param_list(self):
            return [float(self.params[i]) for i in range(self.params_len)]

    observer_fn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                   ctypes.POINTER(LmIter))

    class LmConfig(ctypes.Structure):
        """The solver configuration, holding the preset's Rust values
        (the generated module's constructors fetch them through the
        FFI). Optional fields read/assign as float-or-None; assign a
        Python callable to `observer` (it receives an LmIter; return
        False to stop the solve)."""
        _fields_ = [
            ("preset", ctypes.c_uint32),
            ("max_iters", ctypes.c_uint32),
            ("min_iters", ctypes.c_uint32),
            ("patience", ctypes.c_uint32),
            ("num_threads", ctypes.c_uint32),
            ("verbose", ctypes.c_bool),
            ("gather_timing", ctypes.c_bool),
            ("abs_precision", fp),
            ("rel_precision", fp),
            ("initial_lambda", fp),
            ("cost_threshold", fp),
            ("lambda_floor", fp),
            ("_gradient_tolerance", COptF),
            ("_parameter_tolerance", COptF),
            ("_predicted_reduction_tolerance", COptF),
            ("_min_diagonal", COptF),
            ("_time_limit_seconds", COptSeconds),
            ("_observer", observer_fn),
            ("_observer_user", ctypes.c_void_p),
        ]

        gradient_tolerance = _opt_property("_gradient_tolerance")
        parameter_tolerance = _opt_property("_parameter_tolerance")
        predicted_reduction_tolerance = \
            _opt_property("_predicted_reduction_tolerance")
        min_diagonal = _opt_property("_min_diagonal")
        time_limit_seconds = _opt_property("_time_limit_seconds")

        @property
        def observer(self):
            return getattr(self, "_observer_py", None)

        @observer.setter
        def observer(self, fn):
            # The CFUNCTYPE object must outlive the config (the
            # classic ctypes GC trap) -- kept as an attribute.
            if fn is None:
                self._observer_py = None
                self._observer_keep = None
                self._observer = observer_fn()
                return

            def trampoline(_user, it):
                r = fn(it.contents)
                return r is None or bool(r)

            self._observer_py = fn
            self._observer_keep = observer_fn(trampoline)
            self._observer = self._observer_keep

    class LmResult(ctypes.Structure):
        """A completed solve: costs, iterations, status, damping, and
        (when gathered) the timing breakdown."""
        _fields_ = [
            ("start_cost", fp),
            ("end_cost", fp),
            ("iterations", ctypes.c_uint32),
            ("accepted_iterations", ctypes.c_uint32),
            ("_status", ctypes.c_int32),
            ("final_lambda", fp),
            ("_timing", LmTiming),
            ("has_timing", ctypes.c_bool),
        ]

        @property
        def status(self):
            return LmStatus(self._status)

        @property
        def timing(self):
            return self._timing if self.has_timing else None

    return {
        "COptF": COptF,
        "COptSeconds": COptSeconds,
        "LmIter": LmIter,
        "LmConfig": LmConfig,
        "LmResult": LmResult,
        "observer_fn": observer_fn,
    }
