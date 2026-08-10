//! Rectangle via the Action API.
//!
//! Same 4-by-2 rectangle as `rectangle_solver`, but constructed
//! through the backend's `Action` enum and applied through a
//! `History` so undo/redo works the same way the GUI toolbar uses.
//! This is the intermediate layer: one level above the raw
//! `Sketch` (you still hand-pick which `Action` variants to push)
//! and one level below the command parser.
//!
//! Run with:
//!   cargo run -r -p arael-sketch-backend --example rectangle_actions
//!
//! Demonstrates:
//! - Constructing `Action` variants directly.
//! - `Action::apply(&mut sketch)` as the canonical mutation path.
//! - Wrapping a batch in a `History` group so undo rewinds it as one.
//! - A round-trip undo / redo at the end to prove history integrity.

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_backend::{Action, CursorState, History};
use arael_sketch_solver::*;

fn main() {
    let mut sketch = Sketch::new();
    let mut history = History::new(&sketch);

    // Group all rectangle construction under one undo-frame so Ctrl+Z
    // rewinds "the rectangle" as a single unit rather than 10+ steps.
    history.begin_group();

    let bottom = push(&mut sketch, &mut history,
        Action::AddLine { p1: vect2d::new(0.0, 0.0), p2: vect2d::new(3.0, 0.1) });
    let right = push(&mut sketch, &mut history,
        Action::AddLine { p1: vect2d::new(3.1, 0.0), p2: vect2d::new(3.0, 2.1) });
    let top = push(&mut sketch, &mut history,
        Action::AddLine { p1: vect2d::new(2.9, 2.0), p2: vect2d::new(0.1, 1.9) });
    let left = push(&mut sketch, &mut history,
        Action::AddLine { p1: vect2d::new(0.0, 2.1), p2: vect2d::new(0.1, 0.1) });

    // Horizontal and vertical constraints -- the action takes a list
    // so a single Action can flag multiple lines at once.
    push_void(&mut sketch, &mut history,
        Action::ApplyHorizontal { lines: vec![bottom, top] });
    push_void(&mut sketch, &mut history,
        Action::ApplyVertical { lines: vec![left, right] });

    // Four corner coincidences. LL21 = "line1.p2 coincident with line2.p1".
    for (a, b) in [(bottom, right), (right, top), (top, left), (left, bottom)] {
        push_void(&mut sketch, &mut history,
            Action::ApplyCoincidentLL21 { a, b });
    }

    // Two length dimensions. AddDimension takes a kind + value + optional
    // parametric expression (None here -- pure numeric dimensions).
    push_void(&mut sketch, &mut history,
        Action::AddDimension {
            kind: DimensionKind::LineLength(bottom),
            value: 4.0, expr: None, derived: false, range: None,
        });
    push_void(&mut sketch, &mut history,
        Action::AddDimension {
            kind: DimensionKind::LineLength(left),
            value: 2.0, expr: None, derived: false, range: None,
        });

    // Pin the bottom-left corner via the LockLineP1 action so the
    // fix is part of the history group and comes back on redo.
    push_void(&mut sketch, &mut history,
        Action::LockLineP1 { line: bottom, pos: vect2d::new(0.0, 0.0) });

    sketch.solve();
    println!("after build: dof={:?}", sketch.cached_dof());
    dump(&sketch, bottom, right, top, left, "built");

    // Undo the whole group (Ctrl+Z in the GUI would do the same).
    let (mut restored, _cursor) = history.undo().expect("undo ok");
    println!("after undo: lines = {}", restored.lines.refs().count());

    // Redo; geometry is restored.
    let (redone, _cursor) = history.redo().expect("redo ok");
    restored = redone;
    dump(&restored, bottom, right, top, left, "redone");
}

/// Push an AddLine action, solve, return the new Ref<Line>.
///
/// Action::apply mutates the sketch and performs its own solve; we
/// snapshot afterwards so the history entry captures the post-solve
/// state.
fn push(sketch: &mut Sketch, history: &mut History, action: Action) -> Ref<Line> {
    let before = sketch.lines.refs().count();
    action.apply(sketch);
    history.push(action, sketch, CursorState::default());
    // The newly-added line is the last one.
    sketch.lines.refs().nth(before).expect("AddLine added a line")
}

/// Push an action that does not produce a Ref we need.
fn push_void(sketch: &mut Sketch, history: &mut History, action: Action) {
    action.apply(sketch);
    history.push(action, sketch, CursorState::default());
}

fn dump(sketch: &Sketch, bottom: Ref<Line>, right: Ref<Line>, top: Ref<Line>, left: Ref<Line>, tag: &str) {
    let b = &sketch.lines[bottom];
    let r = &sketch.lines[right];
    let t = &sketch.lines[top];
    let l = &sketch.lines[left];
    println!("[{tag}] bottom: ({:.3},{:.3})->({:.3},{:.3})",
        b.p1.value.x, b.p1.value.y, b.p2.value.x, b.p2.value.y);
    println!("[{tag}] right:  ({:.3},{:.3})->({:.3},{:.3})",
        r.p1.value.x, r.p1.value.y, r.p2.value.x, r.p2.value.y);
    println!("[{tag}] top:    ({:.3},{:.3})->({:.3},{:.3})",
        t.p1.value.x, t.p1.value.y, t.p2.value.x, t.p2.value.y);
    println!("[{tag}] left:   ({:.3},{:.3})->({:.3},{:.3})",
        l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y);
}
