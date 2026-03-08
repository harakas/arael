# TODO

- **Sketch editor**: Tag constraints owned by dimensions so they can be deleted logically instead of matching by approximate value comparison. Currently `RemoveDimension` in `arael-sketch/examples/editor.rs` searches for the underlying constraint (e.g. `distance_pl`) by floating-point distance match, which is fragile (e.g. signed vs unsigned mismatch for `PointLineDistance` — the constraint stores -5.0 but the dimension stores 5.0).
- **Sketch editor**: Add `MoveDimension` action to cleanly reposition dimension annotations (offset + text_along). Currently dimension dragging uses the `Drag` action which snapshots the entire sketch state. Alternatively, extend `AddDimension` to accept optional offset/text_along so the position can be specified at creation time.
