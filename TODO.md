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
- offset tool
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
- **Document Jacobian feature**: DONE -- documented in lib.rs features, macro docs, README.
- **Degenerate tangent at shared endpoint**: DONE -- TangentLA uses dot-product formulation when tangent point is a shared endpoint (detected via coincident scan). The perpendicular-distance formulation has zero Jacobian at shared endpoints; the dot-product does not.
- **DOF computation ownership in Sketch**: Move async DOF lifecycle into Sketch itself. Mutations kick off async computation internally. `sketch.dof()` blocks and waits if result not ready. GUI reads `sketch.cached_dof` for non-blocking display. Eliminates the external async plumbing (dof_input/dof_output/dof_display, bincode serialize/deserialize copy, poll_dof).
- **arael-sym**: Implement `Mul<E> for f64`, `Mul<E> for i64`, etc. so `2.0 * expr` works (currently only `expr * 2.0` compiles). Same for Add, Sub, Div.
- **arael-sketch**: when rotating arcs the arc radius dimension does not rotate -- DONE
- **arael-sketch**: implement sweep A0 driven/distance L0 driven -- make to current value and driven -- DONE
- **arael-sketch**: arc angles when rotating arc can drift? so I see things like arc sweep 480 degrees -- DONE
- **arael-sketch**: just help: add"Type help full for ..". Add Help button. Open command, expand half sketch, issue help full. -- DONE
- **arael-sketch**: dragging should keep hilight, not hilight others -- DONE
- **arael-sketch**: sometimes we somehow get stuck pasting into cmd input -- ??? browser/wasm issue?
- **arael-sketch**: make language more real so that you can do vector algebra

- **arael-macros**: support general func() syntax -- right now we have to describe all of them which is annoying..
- **arael**: support single struct model+root. right now it does not function.
- **arael**: support global optimization parameters with a triplet block



  Issues / friction                                                                                                                                                   
                                                                                                                                                                      
  1. driven doesn't compose with closest: angle b2 b3 driven closest failed with a parse error. I had to measure first, then manually type the value. driven should   
  work alongside sector-selection keywords.
  2. Symmetry DOF cost is hard to predict. The docs say line-line symmetry removes 2 DOF, with the caveat "may be less when entities share endpoints." In practice,   
  symmetry b3 cl b4 removed only 1 DOF because they share a point on the axis — the equal-length part was redundant. This was my only DOF miscount. It would help if: 
    - The docs explicitly called out this specific case (shared endpoint on axis)
    - Or the tool reported effective DOF removed per constraint (e.g., "Applied symmetry (-1 DOF)" instead of just "Applied symmetry")                                
  3. No DOF-per-constraint feedback. Every constraint just says "Applied X". Reporting the actual DOF delta would make iterative constraint building much faster and  
  eliminate guesswork. This is the single biggest improvement I'd suggest.                                                                                            
  4. No driven closest or driven supplement — more generally, the driven keyword seems to only work as a standalone modifier, not composable with angle sector        
  selectors. Since angle direction ambiguity is common, this combination should be supported.                                                                         
  5. The docs could use a "common pitfalls" section for DOF budgeting with shared topology. The current DOF table gives maximums for independent entities, and the
  caveat is easy to miss. A few concrete examples of reduced DOF (like the shared-point-on-axis case) would save a lot of planning time.                              
              
  Minor suggestions                                                                                                                                                   
              
  - measure is great but I only discovered it late. Promoting it more prominently in the workflow docs (alongside dof) would help.                                    
  - A dof delta mode where you get a running DOF count after each command in a script would be very useful for debugging constraint strategies.
  - The best practices say "prefer relative constraints for DOF=3" — a short checklist of "allowed" vs "forbidden" constraints for DOF=3 sketches would be handy      
  (e.g., no lock, horizontal, vertical, hdistance, vdistance, xangle).   

