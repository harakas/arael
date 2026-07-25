# arael Python g2o pose-graph file I/O: the SE2 subset of arael's
# src/g2o.rs (VERTEX_SE2 / EDGE_SE2; unknown record types are skipped,
# vertex ids must be dense and ordered). Malformed records raise
# ValueError with the 1-based line number.

from .math import vect2d


class Pose2:
    """One 2D pose from a VERTEX_SE2 record."""

    def __init__(self, t, th):
        self.t = t if isinstance(t, vect2d) else vect2d(t)
        self.th = float(th)


class DeltaPose2:
    """One relative SE2 measurement from an EDGE_SE2 record: pose b
    seen from pose a's body frame. `info` is the information-matrix
    upper triangle in file order (I11 I12 I13 I22 I23 I33)."""

    def __init__(self, a, b, dt, dth, info):
        self.a = int(a)
        self.b = int(b)
        self.dt = dt if isinstance(dt, vect2d) else vect2d(dt)
        self.dth = float(dth)
        self.info = tuple(float(v) for v in info)
        if len(self.info) != 6:
            raise ValueError("EDGE_SE2 needs 6 information entries")

    def iso_sqrt_info(self):
        """(wt, wr) when the information matrix is diagonal with equal
        translation entries; None when it is anything else."""
        i11, i12, i13, i22, i23, i33 = self.info
        if (abs(i12) < 1e-9 and abs(i13) < 1e-9 and abs(i23) < 1e-9
                and abs(i11 - i22) < 1e-9):
            return i11 ** 0.5, i33 ** 0.5
        return None


class Dataset2:
    """A 2D pose graph: poses and the relative measurements between
    them."""

    def __init__(self):
        self.poses = []
        self.deltas = []

    @classmethod
    def parse(cls, text):
        """Parse a 2D pose graph from .g2o text."""
        ds = cls()
        for lineno, line in enumerate(text.splitlines(), 1):
            f = line.split()
            if not f:
                continue
            try:
                if f[0] == "VERTEX_SE2":
                    if int(f[1]) != len(ds.poses):
                        raise ValueError("vertex ids must be dense and ordered")
                    ds.poses.append(Pose2((float(f[2]), float(f[3])),
                                          float(f[4])))
                elif f[0] == "EDGE_SE2":
                    ds.deltas.append(DeltaPose2(
                        int(f[1]), int(f[2]), (float(f[3]), float(f[4])),
                        float(f[5]), [float(v) for v in f[6:12]]))
            except (IndexError, ValueError) as e:
                raise ValueError("g2o line %d: %s" % (lineno, e)) from None
        for d in ds.deltas:
            if d.a >= len(ds.poses) or d.b >= len(ds.poses):
                raise ValueError("measurement references pose %d / %d of %d"
                                 % (d.a, d.b, len(ds.poses)))
        return ds

    @classmethod
    def load(cls, path):
        """Read a 2D pose graph from a .g2o file."""
        with open(path, "r") as f:
            return cls.parse(f.read())
