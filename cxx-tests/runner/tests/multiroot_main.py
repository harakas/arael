# Both roots of the multi-root fixture from ONE interpreter: separate
# modules share the package and the cdylib ($ARAEL_CAPI). Prints the
# same "name value" lines as multiroot_main.cpp.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "mr", "python"))

from cxx_mr import decay, line  # noqa: E402
from cxx_mr.arael.math import vect2f  # noqa: E402


def p(n, v):
    print("%s %.17e" % (n, v))


ln = line.Line()
for i in range(1, 5):
    ob = ln.obs.push()
    ob.x = float(i)
    ob.y = 3.0 * i + (0.25 if i % 2 == 0 else -0.25)
lc = line.LmConfig()
lc.max_iters = 30
lr = ln.solve_dense(lc)
p("line_status", float(int(lr.status)))
p("line_end", lr.end_cost)
p("line_k", ln.k)

dc_model = decay.Decay()
for i, t in enumerate([0.5, -1.5, 2.0]):
    c = dc_model.cells.push()
    c.t = t
    c.w = 1.0 + i
# A tie is not generic and names the generic Cell: its refs are CellRef
# handles, its math value a vect2f.
tie = dc_model.ties.push()
tie.a = dc_model.cells.ref_at(0)
tie.b = dc_model.cells.ref_at(2)
tie.d = 0.75
tie.at = vect2f(1.5, -2.5)
p("tie_a_is_cell0", 1.0 if tie.a == dc_model.cells.ref_at(0) else 0.0)
p("tie_b_is_cell2", 1.0 if tie.b == dc_model.cells.ref_at(2) else 0.0)
p("tie_at_x", tie.at.x)
p("tie_at_y", tie.at.y)
dc = decay.LmConfig()
dc.max_iters = 30
dr = dc_model.solve_dense(dc)
p("decay_status", float(int(dr.status)))
p("decay_end", dr.end_cost)
for i in range(3):
    p("cell%d" % i, dc_model.cells[i].v)
