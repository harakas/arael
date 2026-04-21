//! Rectangle via the Sketch Solver API.
//!
//! Drives `arael_sketch_solver` directly -- no command parser, no
//! undo/redo, no GUI. Builds four lines, flags two horizontal and two
//! vertical, stitches the corners together with `CoincidentLL21`
//! constraints, fixes the bottom-left corner, sets two length flags,
//! and runs the Levenberg-Marquardt solver. This is the lowest-level
//! entry point into arael-sketch; use it when you want the solver
//! but not the scripting layer.
//!
//! Run with:
//!   cargo run -r -p arael-sketch --example rectangle_solver

use arael::model::{CrossBlock, Param};
use arael::vect::vect2d;
use arael_sketch_solver::*;

fn main() {
    let mut sketch = Sketch::new();

    // Four lines, deliberately a bit off so the solver has work to do.
    let bottom = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.1));
    let right  = sketch.add_line(vect2d::new(3.1, 0.0), vect2d::new(3.0, 2.1));
    let top    = sketch.add_line(vect2d::new(2.9, 2.0), vect2d::new(0.1, 1.9));
    let left   = sketch.add_line(vect2d::new(0.0, 2.1), vect2d::new(0.1, 0.1));

    // Horizontal / vertical flag constraints on the line struct itself.
    sketch.lines[bottom].constraints.horizontal = true;
    sketch.lines[top].constraints.horizontal = true;
    sketch.lines[left].constraints.vertical = true;
    sketch.lines[right].constraints.vertical = true;

    // Corner coincidences: line a's p2 == line b's p1, around the loop.
    // LL21 = "line1.p2 coincident with line2.p1".
    for (a, b) in [(bottom, right), (right, top), (top, left), (left, bottom)] {
        sketch.coincident_ll21.push(CoincidentLL21 {
            a, b, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
    }

    // Pin the bottom-left corner so the sketch cannot translate.
    sketch.lines[bottom].p1 = Param::fixed(vect2d::new(0.0, 0.0));

    // Length dimensions expressed as flag constraints on the line.
    sketch.lines[bottom].constraints.has_length = true;
    sketch.lines[bottom].constraints.length = 4.0;
    sketch.lines[left].constraints.has_length = true;
    sketch.lines[left].constraints.length = 2.0;

    // Solve every constraint simultaneously.
    let result = sketch.solve();
    println!("solver: iters={}, cost={:.6} -> {:.6}",
             result.iterations, result.start_cost, result.end_cost);

    // Read the optimised geometry back.
    let b = &sketch.lines[bottom];
    let r = &sketch.lines[right];
    let t = &sketch.lines[top];
    let l = &sketch.lines[left];
    println!("bottom: ({:.3},{:.3}) -> ({:.3},{:.3})",
             b.p1.value.x, b.p1.value.y, b.p2.value.x, b.p2.value.y);
    println!("right:  ({:.3},{:.3}) -> ({:.3},{:.3})",
             r.p1.value.x, r.p1.value.y, r.p2.value.x, r.p2.value.y);
    println!("top:    ({:.3},{:.3}) -> ({:.3},{:.3})",
             t.p1.value.x, t.p1.value.y, t.p2.value.x, t.p2.value.y);
    println!("left:   ({:.3},{:.3}) -> ({:.3},{:.3})",
             l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y);
}
