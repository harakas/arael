# TODO

- **Sketch editor**: Tag constraints owned by dimensions so they can be deleted logically instead of matching by approximate value comparison. Currently `RemoveDimension` in `arael-sketch/examples/editor.rs` searches for the underlying constraint (e.g. `distance_pl`) by floating-point distance match, which is fragile (e.g. signed vs unsigned mismatch for `PointLineDistance` — the constraint stores -5.0 but the dimension stores 5.0).
- **Sketch editor**: Add `MoveDimension` action to cleanly reposition dimension annotations (offset + text_along). Currently dimension dragging uses the `Drag` action which snapshots the entire sketch state. Alternatively, extend `AddDimension` to accept optional offset/text_along so the position can be specified at creation time.
- derived dimensions -- DONE
- hiding of intermediary PcN points -- DONE
- dragging hide cost != 0
- investigate arc negative radiuses -- DONE (ccw flag + sweep_sign + negative radius rejection)
- implement elliptic arcs
- rect tools etc
- circle creaing tools etc
- mirror tool
- fillet tool
- trim tool
- split tool
- scale tool
- mirror tool
- various circle tools
- text placement
- polygon tool
- **Duplicate constraint check**: `symmetry_pp` (point symmetry) skips duplicate detection because `resolve_as_point` creates helper points before we can check — need to compare semantic endpoints, not Ref<Point> values
- **Redundancy warning**: DONE -- constraints now checked for DOF reduction, rejected if redundant. Use `force` to override.
- Way to get the Jacobian for the system with constraints identifiable for more efficient SVD analysis of DOF in arael-sketch -- DONE (`#[arael(root, jacobian)]` + `#[arael(constraint_index)]`)
- **Document Jacobian feature**: Add documentation for `#[arael(root, jacobian)]`, `#[arael(constraint_index)]`, `calc_jacobian()`, `Jacobian<T>`/`JacobianRow<T>` types, and `ExtendedModel::extended_jacobian64/32` to crate docs and README. Include the `jacobian_demo` example in the docs.
- **Degenerate tangent at shared endpoint**: DONE -- TangentLA uses dot-product formulation when tangent point is a shared endpoint (detected via coincident scan). The perpendicular-distance formulation has zero Jacobian at shared endpoints; the dot-product does not.
- **arael-sym**: Implement `Mul<E> for f64`, `Mul<E> for i64`, etc. so `2.0 * expr` works (currently only `expr * 2.0` compiles). Same for Add, Sub, Div.

