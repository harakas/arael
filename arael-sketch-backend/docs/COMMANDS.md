# Arael Sketch — Command Reference

Arael Sketch is a 2D parametric constraint-based sketch editor. You build geometry (lines, arcs, circles, points), apply geometric constraints (horizontal, parallel, coincident, tangent, etc.), and add dimensions to control sizes and distances. The solver automatically adjusts geometry to satisfy all constraints and dimensions in real time.

This is the complete command reference. Commands can be entered in the command panel (press `/` to open, `Escape` to close) or sent programmatically via the MCP server. Commands can be chained with `;`.

## Coordinate System

The coordinate system uses standard math convention: **X-axis points right, Y-axis points up**. Positive angles are counter-clockwise. This is NOT screen convention (where Y points down).

## Entity Parameters and DOF

Each entity type has a fixed number of degrees of freedom (DOF) — parameters the solver can adjust:

| Entity | DOF | Parameters |
|--------|-----|------------|
| Point  | 2   | x, y |
| Line   | 4   | p1.x, p1.y, p2.x, p2.y |
| Arc    | 5   | center.x, center.y, radius, start_angle, end_angle |
| Circle | 3   | center.x, center.y, radius (angles fixed at 0 and 2pi) |
| Ellipse | 5  | center.x, center.y, radius, radius_b, rotation (angles fixed) |
| Elliptic arc | 7 | center.x, center.y, radius, radius_b, rotation, start_angle, end_angle |

The serialized parameter count differs from the DOF: circles carry 4
parameters and arcs 6, because `radius_b` is always a parameter and an
internal constraint pins it to `radius` for non-ellipses.

Each constraint removes 1 or more DOF. A fully constrained sketch has DOF 0. Use `dof` to check, `dof analyze` to see which entities can still move.

### Constraint DOF Table

| Constraint | DOF removed |
|------------|-------------|
| `horizontal` / `vertical` | 1 |
| `parallel` | 1 |
| `perpendicular` | 1 |
| `equal` (length or radius) | 1 |
| `collinear` | 2 |
| `tangent` | 1 |
| `point_on` (point on line/arc) | 1 |
| `midpoint` | 2 |
| `coincident` (point-point, endpoint-endpoint) | 2 |
| `concentric` | 2 |
| `on_normal` | 1 |
| `image` (pattern copy of a line / point / arc) | 2 per point, 1 per radius / angle row (masked) |
| `symmetry` (point-point about line) | 2 |
| `symmetry` (line-line about line) | 2 |
| `symmetry` (arc-arc about line) | 3 |
| `lock` (point) | 2 |
| `lock` (line endpoint) | 2 |
| Dimension (`length`, `radius`, `sweep`, `angle`, `distance`, `hdistance`, `vdistance`, `xangle`) | 1 |

These are maximum DOF reductions for independent entities. When entities share endpoints (via coincident constraints), effective removal may be less. Always verify with `dof` after applying constraints.

**Line-line symmetry** constrains that two lines are mirror images about an axis, but does not bind their endpoints to specific positions — endpoints can still slide along the lines. To fully constrain symmetric endpoint positions, add coincident or lock constraints on the endpoints separately.

### DOF Budgeting Example

Plan your constraints by counting DOF. Example: fully constrained triangle.

```
# 3 lines: 3 x 4 = 12 DOF
# Auto-connect adds 3 coincident constraints: -6 DOF (12 - 6 = 6)
s0 = add_line 0,0 5,0
s1 = add_line 5,0 3,4
s2 = add_line 3,4 0,0

# Fix sizes: 2 lengths = -2 DOF, 1 angle = -1 DOF
length s0 5
length s1 4
angle s0 s1 60 closest

# Fix position and orientation: lock = -2, horizontal = -1
lock s0.p1 0,0
horizontal s0

# Total: 12 - 6 - 2 - 1 - 2 - 1 = 0 DOF (fully constrained)
```

Note: when you create lines with matching endpoints (e.g., L1 starts where L0 ends), the editor automatically creates coincident constraints. You don't need to add them manually.

## Entity References

Entities are referenced by name: `L0`, `L1` (lines), `P0`, `P1` (points), `A0`, `A1` (arcs/circles).

Endpoints use dot notation:
- Lines: `L0.p1`, `L0.p2`
- Arcs: `A0.center`, `A0.start`, `A0.end`
- Points: `P0` (shorthand for `P0.pos`)

## Coordinates

Coordinates can be specified as:

| Format | Example | Description |
|--------|---------|-------------|
| `x,y` | `5,3` | Absolute position |
| `@dx,dy` | `@0,3` | Relative to cursor |
| `cursor` | `cursor` | Current cursor position |
| Endpoint ref | `L0.p2` | Current position of an endpoint |
| Session var | `p` | A variable set with `let` |
| Geo function | `midpoint(L0)` | Result of a geometric function |
| Expression | `width,height` | Expressions using params/properties |
| Vector expr | `L0.p2+normal(L0)*3` | Arithmetic on coordinates |

## Expressions

Anywhere a numeric value is expected, an expression can be used. Expressions can reference:

- **Entity properties**: `L0.p1.x`, `L0.p2.y`, `L0.length`, `L0.angle`, `A0.radius`, `A0.diameter`, `A0.sweep`, `A0.start_angle`, `A0.end_angle`, `A0.center.x`, `A0.start.x`, `A0.end.y`, `P0.x`, `P0.y`
- **Dimension values**: `d0`, `d1`
- **User parameters**: `width`, `height`
- **Session variables**: variables set with `let`
- **Math functions**: `sqrt()`, `sin()`, `cos()`, `atan2()`, `abs()`, `pi`

## Geometry Creation

```
add_line x1,y1 x2,y2        Create a line between two points
add_line x1,y1 x2,y2 x3,y3  Multi-segment: creates connected lines (any number of points)
add_line x2,y2               Continue from last endpoint (single-point chaining)
add_line @dx,dy              Continue with relative offset
add_rect x1,y1 x2,y2        Rectangle from two opposite corners (4 lines)
add_rect3 p1 p2 p3           Rectangle from 3 consecutive corners (4th computed)
add_rectcenter cx,cy px,py   Rectangle from center and one corner
add_point x,y [nocursor]     Create a free point
add_circle cx,cy radius      Create a circle (full arc)
add_circle2 p1 p2            Circle from 2 diametrically opposite points
add_circle3 p1 p2 p3         Circle from 3 points on circumference
add_circle2t L0 L1 radius    Circle tangent to 2 lines with given radius
add_circle3t L0 L1 L2        Circle tangent to 3 lines (incircle)
add_arc x1,y1 x2,y2 xm,ym  Create an arc through start, end, midpoint [driven]
offset_line L0 distance      Create a parallel line offset by distance (alias: offset)
fillet L1 L2 r               Round the shared corner of L1 and L2 with a tangent arc of radius r
fillet L1.pN r               Same, naming the corner endpoint directly
fillet L1 L2 L3.pN ... r     Multiple corners at once (any mix of two-line and endpoint forms); last token is the radius
chamfer L1 L2 d              Bevel the shared corner at distance d from the corner
chamfer L1.pN d              Same, naming the corner endpoint directly
chamfer L1 L2 L3.pN ... d    Multiple corners at once
split L0 x,y [r]             Cut L0 at the intersections bracketing x,y; all pieces survive
split L0 by L1 L2            Cut L0 at every intersection with the named cutters
trim L0 x,y [r]              Delete the span of L0 containing x,y
trim L0 by L1 L2             Delete the span of L0 between the two cutters
trim L0 by L1 forward        Delete from L1's crossing to the p2/end side
scale L0 A0 about P0 2       Uniform scale about a point; center is an endpoint or coordinate
scale selection about 0,0 0.5  Scale the current selection
```

### Auto-Coincident

When creating geometry (`add_line`, `add_circle`, `add_arc`, `add_earc`, `add_earc3`, `add_earc_center`), endpoints that are within 1e-3 distance of existing endpoints are automatically connected with coincident constraints. Lines snap to line endpoints, points, and arc endpoints. Circles and arcs snap their center (and start/end for arcs) to line endpoints, other arc endpoints, and points.

```
add_line 0,0 5,0              L0
add_line 5,0 5,3              L1 — auto-connects L1.p1 to L0.p2
add_circle 5,0 1              A0 — auto-connects A0.center to L0.p2
add_circle 0,0 2              A1 — auto-connects to L0.p1
add_line 0,0 5,3              L2 — auto-connects L2.p2 to A0.center
```

The output shows what was connected, with the constraint id: `Added L1: ... [connected: L1.p1=L0.p2 (C3)]`

Append `noconnect` to suppress auto-connection:
```
add_line 5,0 5,3 noconnect      No auto-coincident
add_circle 5,0 1 noconnect      No auto-coincident
add_arc 0,0 5,0 2,3 noconnect   No auto-coincident
```

When 3+ endpoints meet at the same point, auto-connect creates multiple coincident constraints. Some may be redundant (e.g., if A=B and B=C, then A=C is implied). Redundant constraints are rejected by the DOF check -- this is normal.

### Auto-Tangent

When a new line or arc shares an endpoint with an existing arc (via auto-coincident), the system checks if the geometry is already tangent at that point. If the tangent directions are within 1 degree and the constraint cost increase is below 1e-6, a tangent constraint is automatically applied.

```
add_line 0,0 5,0
add_arc 5,0 5,5 7.5,2.5     A0 — auto-tangent with L0 if directions match
```

The output shows: `Added A0 [connected: A0.start=L0.p2 (C3)] [tangent: L0.tangent.A0 (C4)]`

Auto-tangent works for line-arc and arc-arc connections at shared endpoints.

Append `notangent` to suppress auto-tangent (auto-coincident still applies):
```
add_arc 5,0 5,5 7.5,2.5 notangent
```

`noconnect` implies `notangent` (no connections means no tangent either).

### Quiet Mode

Entities marked as `quiet` suppress visual clutter:
- Intrinsic dimensions (radius, length, sweep) are hidden unless the entity or dimension is selected
- Arc/circle/ellipse center points are hidden unless a constraint references the center or it is selected

Append `quiet` during creation:
```
add_line 0,0 5,0 quiet
add_circle 0,0 5 quiet
add_earc 0,0 5,0 3 1 0 quiet
```

Toggle quiet on existing entities:
```
quiet L0                     Toggle quiet on/off
quiet L0 on                  Set quiet
quiet L0 off                 Remove quiet
quiet EA0 EA1 L0             Multiple entities
```

`info` and `list` show `[quiet]` marker.

### Drag

Drag entities or endpoints to a new position. The sketch relaxes under constraints after the drag.

```
drag L0.p1 5,3               Drag line endpoint to absolute position
drag L0.p2 @0,3              Drag line endpoint by relative offset
drag L0 @2,0                 Drag entire line body
drag A0.center 0,0           Drag arc center
drag A0.start @-1,2          Drag arc start point
drag A0 @0,1                 Drag entire arc body (locks shape)
drag P0 3,3                  Drag standalone point
```

If the drag results in unsatisfiable constraints, the sketch reverts to its pre-drag state and an error is reported.

### Construction Lines

Entities marked as construction are reference geometry — drawn in a distinct color with dashdot style.

Append `constr` during creation:
```
add_line 0,0 5,0 constr
add_circle 0,0 5 constr
```

Toggle construction on existing entities:
```
constr L0                     Toggle construction on/off
constr L0 on                  Set construction
constr L0 off                 Remove construction
```

The X key in the GUI toggles construction state on selected entities.

`info` shows `[constr]` flag. `list` shows `[constr]` marker. `list constr` shows only construction entities.

### Line Chaining

When `add_line` is given only one coordinate, it starts from the last created endpoint. With 3+ coordinates, it creates multiple connected segments in one command:

```
add_line 0,0 5,0             L0: (0,0)-(5,0)
add_line @0,3                L1: (5,0)-(5,3)   single-point chaining
add_line @-5,0               L2: (5,3)-(0,3)
add_line L0.p1               L3: (0,3)-(0,0)   close the rectangle

add_line 0,0 1,0 2,1 3,0    Creates L0, L1, L2 as connected segments
add_line 0,0 @1,0 @0,1 @-1,0  Relative coords work for multi-segment too
```

### Rectangles

Create rectangles as 4 connected lines with constraints. All rectangle commands support these optional trailing keywords (in any order):

- `noconnect` — skip auto-coincident with external geometry (internal corners still share coordinates)
- `noconstraint` — create geometry only, no constraints
- `hv` — use horizontal/vertical constraints instead of perpendicular/parallel (not available for `add_rect3`)
- `driven` — add driven length dimensions on two adjacent sides
- `strict` — error if any constraint/dimension fails to apply (default: warn and continue)

`noconstraint` conflicts with `hv`, `driven`, and `strict`.

Session names `_0`..`_3` are set to the four line names.

```
# Two-point axis-aligned rectangle (opposite corners)
add_rect 0,0 5,3                          Default: perpendicular + 2 parallel
add_rect 0,0 5,3 hv                       Horizontal/vertical constraints
add_rect 0,0 5,3 driven                   + driven length dimensions on 2 sides
add_rect 0,0 @5,3                         Relative second corner

# Three-point rectangle (arbitrary orientation)
add_rect3 0,0 5,0 5,3                     p1-p2 is one side, p2-p3 adjacent side
add_rect3 0,0 @5,0 @0,3                   Relative coordinates

# Rectangle from center and corner
add_rectcenter 2.5,1.5 0,0                Center + corner (axis-aligned)
add_rectcenter 2.5,1.5 0,0 hv driven      With hv + driven dimensions
```

Example with variable capture:
```
bot, right, top, left = add_rect 0,0 5,3 hv driven
length bot 10                              Update width
```

### Driven Dimensions on Lines and Circles

`add_line`, `add_circle`, and `add_arc` support the `driven` keyword to automatically create driven length/radius dimensions:

```
add_line 0,0 5,0 driven                   Line with driven length dimension
add_line 0,0 5,0 5,3 driven               Multi-segment: each segment gets a dimension
add_circle 0,0 3 driven                   Circle with driven radius dimension
add_arc 0,0 5,0 2.5,2 driven             Arc with driven radius + sweep dimensions
```

### Circle Construction Tools

```
# Circle from 2 diametrically opposite points
add_circle2 0,0 10,0                       Center at midpoint, radius = half distance
add_circle2 0,0 10,0 driven               With driven radius dimension

# Circle from 3 points on circumference
add_circle3 0,0 5,0 2.5,4                  Circumscribed circle through 3 points

# Circle tangent to 2 lines (fillet)
add_circle2t L0 L1 2                       Circle with r=2 tangent to both lines
add_circle2t L0 L1 2 driven               With driven radius dimension
add_circle2t L0 L1 2 noconstraint          No tangent constraints

# Circle tangent to 3 lines (incircle)
add_circle3t L0 L1 L2                      Incircle of triangle formed by 3 lines
add_circle3t L0 L1 L2 driven              With driven radius dimension
```

`add_circle2t` and `add_circle3t` support: `noconnect`, `noconstraint`, `driven`, `strict`.
`add_circle2` and `add_circle3` support: `noconnect`, `nocursor`, `driven`.

**Placement rule for `add_circle2t` / `add_circle3t`:** The tangent circle is placed so its tangent points fall on the actual line segments (not their infinite extensions). If no placement touches all segments, or if multiple placements do, the command errors out. Control which sector the circle goes into by adjusting line segment lengths.

### Ellipses

Create ellipses with `add_ellipse`. Ellipses have two radii (semi-major `rx`, semi-minor `ry`) and a rotation angle.

```
add_ellipse cx,cy rx ry rotation_deg [noconnect] [nocursor] [driven]
add_ellipse 0,0 5 3 45                        Ellipse rotated 45 degrees
add_ellipse 0,0 5 3 0 driven                  With driven rx and ry dimensions
```

Ellipses are named with the `EA` prefix (e.g. `EA0`, `EA1`) to distinguish them from circular arcs (`A0`, `A1`).

Ellipse-specific dimensions:
```
radius EA0 5                 Set semi-major axis (same as circle radius)
radius_b EA0 3               Set semi-minor axis (ellipses only)
radius_b EA0 driven          Driven minor axis dimension
```

Ellipse parameters are accessible as `EA0.radius`, `EA0.radius_b`, `EA0.rotation` in expressions and `print`.

### Elliptic arcs

Partial elliptic arcs (open ellipse segments). Three creation styles:

**SVG endpoint-based** (inspired by SVG arc commands):
```
add_earc p1 p2 rx ry rot_deg [large] [cw] [noconnect] [notangent] [nocursor] [driven]
add_earc 0,0 5,0 3 1 0                   Small CCW arc from (0,0) to (5,0)
add_earc 0,0 5,0 3 1 0 large             Large arc variant
add_earc 0,0 5,0 3 1 0 cw                Clockwise arc
add_earc 0,0 5,0 3 1 45 large cw driven  All options combined
```

**Three-point with radii** (midpoint determines which arc):
```
add_earc3 p1 p2 pmid rx ry [noconnect] [nocursor] [driven]
add_earc3 0,0 5,0 2,2 3 1                Arc passing near (2,2)
```

**Center-based** (direct parameterization, angles in degrees):
```
add_earc_center cx,cy rx ry rot_deg start_deg end_deg [cw] [noconnect] [notangent] [nocursor] [driven]
add_earc_center 0,0 3 1 45 0 90          Quarter elliptic arc
```

**Tangent-defined** (endpoints + tangent directions + curvature weight):
```
add_earc_tangent p1 t1 p2 t2 [bulge] [noconnect] [notangent] [nocursor] [quiet] [driven]
add_earc_tangent 0,0 1,0 5,3 0,1          Horizontal start, vertical end (w=1, circular)
add_earc_tangent 0,0 1,0 5,3 0,1 2        Tighter curve at start (w=2, elliptic)
add_earc_tangent @cursor @tangent 10,3 0,1  Chain from previous entity
```

The `w` parameter (default 1.0) controls curvature at the start point relative to the circular baseline:
- `w=1`: circular arc
- `w>1`: tighter curve at start (higher curvature, more elliptic)
- `w<1`: gentler curve at start

**Tangent-chaining** (continues from previous entity's cursor and tangent):
```
add_earc_rtangent p2 t2 [bulge] [noconnect] [notangent] [nocursor] [quiet] [driven]
add_line 0,0 5,0
add_earc_rtangent 10,3 0,1 0.5            Continues smoothly from line
add_earc_rtangent 15,0 1,0 0.5            Chains from previous arc
```

Shorthand for `add_earc_tangent @cursor @tangent p2 t2 [bulge]`. Requires a previous line or arc to provide cursor and tangent direction.

All elliptic arc commands return the arc reference (e.g. `EA0`). With `driven`, both `rx` and `ry` dimensions are added.

### @cursor and @tangent

`@cursor` returns the current cursor position (same as `cursor`). `@tangent` returns the tangent direction vector at the cursor point from the last created line or arc. These enable smooth curve chaining:

```
add_line 0,0 5,0                               L0, cursor at (5,0), tangent (1,0)
add_earc_tangent @cursor @tangent 10,3 0,1     Continues smoothly from L0
add_earc_tangent @cursor @tangent 15,0 1,0     Continues from previous arc
```

## Deletion

`delete` is the single removal command. It handles entities, named
constraints, dimensions, and the relational (multi-argument) form
of constraint removal. Every removal path goes through `delete` —
there is no separate `remove_constraint` or `remove_dim`.

### By name

```
delete L0                    Delete a line  (L0, L1, ...)
delete P0                    Delete a point (P0, P1, ...)
delete A0                    Delete a circular arc or circle (A0, A1, ...)
delete EA0                   Delete an ellipse or elliptic arc (EA0, EA1, ...)
delete C3                    Delete a named constraint (parallel, coincident,
                             tangent, midpoint, symmetry, ...)
delete CL0H                  Delete a synthetic line-flag constraint
                             (CL<line>H for horizontal, CL<line>V for vertical)
delete d0                    Delete a dimension (d0, d1, ...)
delete selection             Delete everything selected, as one batch
                             (one undo step); a selected meta-constraint
                             dissolves, its geometry stays
```

Use `list` or `info <entity>` to discover the `C<n>` / `d<n>` names
to pass here. Dimension-managed constraints (e.g. the `distance_pp`
coupling behind a `distance P0 P1 5` dimension) are reached through
the backing dimension name, not through a `C<n>` — `delete d0`
removes both the dimension and its underlying constraint.

### Relational (multi-argument) form

When a constraint hasn't got a handy `C<n>` yet (or the user just
wants to describe it by the entities involved), pass the entities
and the constraint type:

```
delete L0 horizontal                  Drop the horizontal flag from L0
delete L0 vertical                    Drop the vertical flag from L0
delete L0 L1 parallel                 Drop a parallel constraint between L0 and L1
delete L0 L1 perpendicular            Drop a perpendicular constraint (alias: perp)
delete L0 L1 equal                    Drop an equal-length constraint (alias: equal_length)
delete A0 A1 equal_radius             Drop an equal-radius constraint
delete L0 L1 collinear                Drop a collinear constraint
delete L0 A0 tangent                  Drop a line-arc tangent
delete A0 A1 tangent                  Drop an arc-arc tangent
delete A0 A1 concentric               Drop a concentric constraint
delete L0.p2 L1.p1 coincident         Drop a point-point coincidence
delete P0 L0 point_on                 Drop a point-on-line constraint
delete L0.p1 A0 point_on              Drop a line-endpoint-on-arc constraint
delete A0.center L0 point_on          Drop an arc-center-on-line constraint
delete P0 L0 P1 symmetry              Drop a point-point-about-line symmetry
delete L0 L1 L2 symmetry              Drop a line-line-about-line symmetry
delete P0 L0 midpoint                 Drop a point-at-midpoint constraint
delete L0.p1 L1 midpoint              Drop a line-endpoint-at-midpoint
delete L0.p1 lock                     Unlock an endpoint  (also: delete P0 lock)
```

The constraint-type token comes last. Either ordering of the two
entities is accepted.

### Cascade on entity deletion

Deleting an entity also deletes every constraint and dimension that
references it — you don't need to tear them down by hand first.
The response message lists each cascaded removal so you can see
exactly what went away with the entity:

```
> delete L0
Deleted line L0
  cascade:
    C1: coincident L1.p1 L0.p2
    CL0H: horizontal L0
    d0: length L0 = 4
```

What cascades:

- **Flag constraints** on the entity (`horizontal`, `vertical`, lock status).
- **Length / xangle** constraints on the entity.
- **Coincident / midpoint / point-on / tangent / symmetry / parallel
  / perpendicular / equal / collinear / concentric / equal-radius**
  constraints whose endpoints touch the entity.
- **Dimensions** whose kind references the entity (for a line: its
  length dim, any point-point / point-line distance involving its
  endpoints, any angle involving it, any line-line distance paired
  with it, etc.).
- **Helper points** that the solver created internally for complex
  coincidences involving the entity. These are invisible to the
  user so they don't appear in the cascade list, but they are
  cleaned up.

Naming shown in the cascade list matches `list` output: `C<n>` for
numbered constraints, `CL<line>H` / `CL<line>V` for flag
constraints, `d<n>` for dimensions, and the natural phrasing
(`length L0 = 4`, `lock L0.p1`) for entity-bound constraints that
don't have their own numeric name.

### Undo

`delete` is a single atomic action in the undo stack — `undo`
restores the entity together with every cascaded constraint and
dimension in one step.

## Constraints

```
horizontal L0 [L1 ...]      Make lines horizontal
vertical L0 [L1 ...]        Make lines vertical
parallel L0 L1               Make two lines parallel
parallel L0 EA0              Align ellipse major axis with a line
parallel EA0 EA1             Align two ellipses' major axes (circular arcs rejected)
perpendicular L0 L1          Make two lines perpendicular (alias: perp)
equal L0 L1                  Equal length (lines) or equal radius (arcs)
collinear L0 L1              Make two lines collinear
tangent L0 A0                Tangent: line-arc (line must be first argument)
tangent A0 A1                Tangent: arc-arc (either order)
coincident L0.p2 L1.p1       Coincident: any endpoint pair
concentric A0 A1             Concentric arcs
on_normal L1.p2 L0.p2        L1.p2 lies on the normal of L0 at L0.p2 (the perpendicular foot)
on_normal A1.start A0.start  A1.start lies on the normal of A0 at A0.start (a circle's radial ray, an ellipse's true normal)
midpoint P0 L0               Point at midpoint of line
midpoint L0.p1 L1            Line endpoint at midpoint of another line
midpoint P0 A0               Point at angular midpoint of arc
midpoint L0.p1 A0            Line endpoint at angular midpoint of arc
symmetry L0 L1 L2            Lines L0,L2 symmetric about L1
symmetry P0 L0 P1            Points P0,P1 symmetric about L0
symmetry A0 L0 A1            Arc centers symmetric + equal radius (3 DOF removed)
symmetry L0.p1 L1 L2.p1     Endpoints symmetric about L1
symmetry A0.center L0 A1.center  Any endpoint ref works (P0, L0.p1, A0.center, A0.start, A0.end)
mirror L0 L1 about L2        Mirror entities about a line (creates copies + symmetry constraints)
mirror selection about L0    Mirror selected entities
point_on P0 L0               Point on line
point_on L0.p1 L1            Line endpoint on another line
point_on L0.p1 A0            Line endpoint on arc
point_on A0.center L0        Arc center on line
```

Every action runs through one validation gate, on every path (GUI, commands, MCP):

- **Logical conflicts.** Contradictions (horizontal on a vertical line, including transitive chains through parallel, collinear and perpendicular links), duplicates, and self-references are rejected.
- **Degenerate geometry.** Zero-length lines, zero-radius circles/ellipses, and arcs through collinear points are rejected at creation. Tangents on zero-length lines or concentric arcs are rejected.
- **DOF check.** Constraint and dimension commands must reduce DOF. If a constraint doesn't reduce DOF, it is rejected -- the constraint is either redundant or already implied by existing constraints.

Append `force` to skip the DOF check only; conflicts and degenerate geometry stay rejected:

```
equal L0 L1 force            Skip DOF check
```

Using `force` results in an overconstrained sketch -- redundant constraints make it harder to modify the sketch later because changing one dimension may conflict with the redundant constraint. Prefer rethinking the constraint strategy over using `force`.

### Dry-Run with `explain`

`explain <cmd> [args]` runs a constraint or dimension command as a dry-run: it evaluates whether the inner command would be accepted or rejected (including the blocker analysis for DOF-rejections) and then restores the sketch to its pre-command state. Nothing is committed, no history entry is added.

```
explain perpendicular L0 L1
explain hdistance L0.p2 L1.p2 3
explain length L0 5
```

Output mirrors a normal command run:
- Accepted: `'<cmd>': accepts -- <normal success message>`
- Rejected: `'<cmd>': rejects -- Constraint rejected: ... Blocked by one of: C1 (...), d0 (...)`

Useful for probing why a constraint would fail before committing to it, or for tooling (MCP, scripts) that wants to surface blocker info without mutating sketch state.

### Removing Constraints

Every constraint has an auto-assigned name. Vec-stored constraints
get a sequential `C<n>` name (`C1`, `C2`, ...) shown by the `list`
command. Flag-style constraints on lines (horizontal, vertical) get
synthetic names of the form `C<entity><flag>`: `CL0H` is "horizontal
on L0", `CL3V` is "vertical on L3". Names are stable across save /
load and undo / redo. Deleting constraints can leave holes in the
numbering; holes are not reused.

```
delete C3                    Remove the constraint named C3
delete CL0H                  Remove horizontal flag from L0
delete CL3V                  Remove vertical flag from L3
info C3                      Show the list line for constraint C3
```

Relational syntax (specify the pair and constraint type):

```
delete L0 horizontal
delete L0 L1 parallel
delete L0 L1 perpendicular
delete L0 L1 equal
delete A0 A1 equal_radius
delete L0 L1 collinear
delete L0 A0 tangent
delete A0 A1 concentric
delete L1.p2 L0.p2 on_normal
delete L0.p2 L1.p1 coincident
delete P0 L0 point_on
delete L0.p1 A0 point_on
delete A0.center L0 point_on
delete P0 L0 P1 symmetry
delete L0 L1 L2 symmetry
delete P0 L0 midpoint
delete L0.p1 L1 midpoint
delete L0.p1 lock
```

Dimension-managed constraints (lengths, radii, angles, distances) are
reached through the dimension name (`delete d0`, `info d0`) and listed
under `list dims`, not `list constraints`. Name-lookup of `C<n>` for
those constraints falls through — use the dimension name instead.

## Mirror

Create mirrored copies of geometry across a mirror line. Endpoint symmetry constraints are added by default. Coincident constraints among source entities are recreated among the mirrored copies.

```
mirror L0 about L1                        Mirror a line
mirror L0 L1 A0 about L2                  Mirror multiple entities
mirror selection about L0                  Mirror selected entities
mirror L0 L1 about L2 noconstraint         No symmetry/coincident constraints
mirror L0 L1 about L2 strict               Error (and undo the mirror) if the result
                                           cannot satisfy all constraints
```

Keywords: `noconstraint` (skip all constraints), `strict` (error and
roll the mirror back if the result cannot satisfy all constraints).
The whole mirror is one undo step.

Variable capture works with `_0`..`_N`:
```
a, b = mirror L0 L1 about L2
```

When mirroring entities that share endpoints (connected via coincident), the mirrored copies also share endpoints, and only one symmetry constraint is created per unique endpoint position (no duplicates).

## Offset

Offset a connected sequence of lines and arcs to one or both sides. Every
result entity is held at the distance from its source by ordinary
constraints and a dimension, and the whole operation is recorded as a
meta-constraint `M<n>` that can be edited afterwards.

```
offset L0 L1 A0 2                 one side, 2 to the left of the chain direction
offset L0 L1 A0 2 right           the other side (also: flip)
offset L0 L1 A0 2 symmetric       both sides at 2
offset L0 L1 A0 2 3               two sides: 2 left, 3 right
offset sequence L0 2              walk from L0 both ways to an end or a branch, then offset
offset selection 2                the current selection, which must be one sequence (or one selected M<n>: edit it)
offset L0 L1 L2 L3 1 inward       closed sequences: inward / outward instead of left / right
offset L0 L1 L2 L3 1 outward round   round the convex corners with an arc of the distance
offset L0 L1 1 symmetric caps round  close the ends: a half circle around each source end (also: caps line)
offset L0 2 nopin                 leave the free ends and tangent joints unpinned
m = offset L0 L1 2                name capture: `_` is the meta, `_0`.. the result entities
```

**The sequence.** Lines and arcs connected end to end (by coincident
endpoints, directly or through a shared point), with no branch inside
it; open or closed (a loop, or a full circle / ellipse on its own). The
chain direction is the first entity's own (`p1 -> p2`, `start -> end`);
`left` is the left-hand side of travel. A set that is not one sequence
is refused, naming the stray or the branch.

**What is created, per source segment:**

| source | result | relation |
|---|---|---|
| line | parallel line at the distance | `parallel` |
| arc / circle | concentric arc, radius +- the distance | `concentric` |
| ellipse / elliptic arc | concentric ellipse, both semi-axes +- the distance | `concentric` + `parallel` (rotation) |

The distance is a `distance` dim on the first result of each run of
tangent joints: the distance carries through a tangent joint, and a
result after a sharp corner has its own dim. So a rounded rectangle's
offset has one dim, a plain rectangle's four.

Consecutive results meet at sharp corners (extended or trimmed to their
intersection) or, where the sources are tangent, at the offset of the
source joint; a coincident joins them. With `round`, a corner the result
goes around (convex on that side) is an arc of the distance centered on
the source corner instead: `coincident` at both ends, `tangent` to both
neighbours, and a `radius` dim; corners the result cuts across stay
sharp. Tangent joints and the free ends of an open sequence are pinned
with `on_normal` so the result has no slide left (`nopin` skips that). A
loop whose joints are all tangent is closed by two pins instead of a
coincident (the ends meet by geometry; a loop of coincidents would be one
equation redundant). The sketch DOF is unchanged by an offset, and every
relation is independent: the result brings exactly as much freedom as
its relations take.

**Caps** close the ends of an open two-sided result. `caps line` is a
line across each end, `coincident` with both results' ends. `caps
round` (symmetric offsets only) is a half circle around each source
end, `coincident` with and `tangent` to both results -- which makes its
radius the distance -- held in place by one `on_normal` pin; the free
ends get no other pins. A closed sequence or a one-sided offset has no
caps.

An ellipse's offset is not an ellipse; the result is the concentric
ellipse with both semi-axes moved by the distance, which is exact at the
axis ends and approximate elsewhere (a few percent of the distance on a
2:1 ellipse). The output says so. A sequence with an elliptic arc tangent
to its neighbour is refused: the approximation cannot keep that joint.

An arc offset inward past its radius has no offset: it vanishes from
that side's result and its neighbours meet directly (a rounded rectangle
offset inward by more than the fillet radius is a sharp rectangle). The
output names it (`vanished: A0 A1`); a later edit that brings it back
rebuilds that side.

Refused, naming the segment: a segment its neighbours' corners would turn
around or shrink to nothing, a chain that doubles back on itself, corners
whose offsets do not meet, an offset where nothing remains.
Distances accept the dimension value forms (`2`, `w/2`, `=w/2`).

**Editing.** The meta-constraint keeps the parameters:

```
offset M0 3                       new distance: the dims are rewritten, the geometry follows
offset M0 flip                    the other side (the geometry is moved across)
offset M0 symmetric | two 2 3 | one   add or remove a side (new entities are reported)
offset M0 round | sharp           corner style: a side whose corners change is rebuilt
offset M0 caps line | round | none   the caps (rebuilt; `one` drops them)
offset M0 nopin | pin             remove or add the pins
info M0 | list metas              what it was made of and what it made
delete M0                         dissolve: the geometry stays, as plain constrained geometry
delete M0 all                     delete the result entities too
```

The record owns what it created. Deleting a result entity, an owned
constraint or an owned dimension, editing or converting an owned
dimension, deleting a source entity, or splitting a result drops the
meta-constraint with a `notice:` line on the command that did it; the
geometry is left as it is. Deleting a pin (`on_normal`) does not drop it.
Adding constraints to the result, dragging, or changing a parameter a
distance expression reads keep it. `info L5` shows the meta-constraint an
entity is the result or source of.

In the GUI the meta-constraint has a marker (two parallel waves) on its
first result: a click selects it and highlights its sources and results
(the panel shows its name and description), a double-click opens it in
the Offset tool, Delete dissolves it.

## Pattern

Copy a set of lines, arcs and points as a circular or rectangular
pattern. Every copy is a rigid image of its source, held there by
`image` constraints (`image L0 -> L4 rotate 90 about P0`), and the whole
operation is recorded as a meta-constraint `M<n>` that can be edited
afterwards. The sketch DOF is unchanged: a copy brings no freedom of its
own.

```
pattern circular L0 A0 P1 about P0 6                 6 instances (the source included) over 360 deg
pattern circular selection about L0.p1 4 partial 90  4 instances over 90 deg, counter-clockwise (negative: clockwise)
pattern circular L0 about A0.center 5 symmetric 120  5 instances over 120 deg centered on the source
pattern rect L0 L1 L2 3 10                           3 instances 10 apart along +x
pattern rect selection 3 10 by 2 5                   a 3 x 2 grid: 10 apart along x, 5 along y
pattern rect L0 4 30 extent along L5                 4 instances spanning 30 along L5 (axis 2 across it, to its left)
pattern rect L0 5 4 symmetric                        5 instances, 2 on each side of the source (even: one less backward)
pattern rect L0 3 -10                                a negative distance reverses the axis
m = pattern rect L0 3 10                             name capture: `_` the meta, `_0`.. the copy entities
```

**Circular**: `about` a point (`P0`) or an endpoint / arc center
(`L0.p1`, `A0.center`; a hidden helper point follows it); then the
quantity and `full` (default), `partial A` or `symmetric A` in degrees.
The copies are rotated about the center (an arc's angles and an
ellipse's rotation turn with the copy).

**Rectangular**: the quantity and distance of axis 1, `by` the quantity
and distance of axis 2 (quantity 1 = off); `along L<n>` takes axis 1
along the line and axis 2 across it to its left (default: +x and +y);
`extent` makes each distance span from the first instance to the last
instead of one step (`spacing`, the default); `symmetric` after an axis
puts instances on both sides of the source; `one` undoes it.

**What is created, per copy:**

| source | image constraint | rows |
|---|---|---|
| line | `image` of the line | its endpoints |
| point | `image` of the point | its position |
| arc / circle / ellipse | `image` of the arc | center, radius (and `radius_b`, rotation), start and end angles |

Every copy keeps the source's coincidences among the copied entities,
so a copy of a polyline is a connected polyline; its `image` rows are
only those the coincidences leave open (a copy parameter is determined
exactly once, every row independent). Other relations among the sources
(parallel, equal, tangent, H/V, dims) are not copied: the rigid image
already implies them. Relations to entities outside the set are not
copied.

Refused: a quantity under 2 (axis 1 of a rectangular pattern under 2
with axis 2 off), a zero distance or angle, the center point or the
direction line inside the set.

**Editing.** The meta-constraint keeps the parameters:

```
pattern M0 8                      new quantity: the copies are made again
pattern M0 partial 90 | full      distribution; a new angle moves the copies in place
pattern M0 3 12 | by 2 6          new quantity / distance on an axis (a distance alone moves in place)
pattern M0 along L2 | noalong | extent | spacing | symmetric | one
pattern M0 about P3               another center (the copies are made again)
info M0 | list metas | select M0 | pattern selection 6
delete M0                         dissolve: the copies stay, still images of the source
delete M0 all                     delete the copies too
```

A distance or angle change rewrites the image constraints and moves
the copies in place (their names stay); anything else -- quantity,
distribution, direction, frame, center, extent / spacing, symmetric --
rebuilds the copies. Deleting a copy entity, an `image` or a recreated
coincidence, a source, the center or the direction line drops the
meta-constraint with a `notice:`; the copies stay as plain constrained
geometry. In the GUI every copy carries the pattern's marker (two by
two squares); an `image` constraint's glyph shows while one of its
entities is selected.

## Dimensions

```
length L0 5                  Numeric literal
length L0 scale              Live expression (re-evaluates every solve)
length L0 2*scale + 1        Compound live expression
length L1 L0.length          Live; L1's length tracks L0's length
length L0 =2*scale           Snapshot: evaluate `2*scale` now, store as a literal
radius A0 1.5                Set arc radius
sweep A0 180                 Set arc sweep angle (degrees)
angle L0 L1 45               Set angle between direction vectors (p1->p2)
angle L0 L1 135 supplement   Use the supplementary angle (180 - default)
angle L0 L1 60 closest       Auto-select sector closest to given value
angle L0 L1 60 acute         Use the smaller current sector
angle L0 L1 120 obtuse       Use the larger current sector
distance L0.p1 L1.p2 5.0     Point-point distance (any endpoint refs)
distance A0.center A1.center 5.0  Arc center to arc center
distance A0.start L1.p2 4.0  Arc start to line endpoint
distance P0 L0 3.0            Point-line distance (point/endpoint first, line second)
distance A0 A1 2.0            Radial distance between two concentric circles/arcs (requires Concentric A0 A1)
distance L0 L1 5.0            Perpendicular distance between two lines; also applies Parallel L0 L1 if not already present. The backing constraint cascades: deleting the Parallel constraint removes the dimension.
distance L0 L1 >= 2.0         Lower-bound range: inter-line gap must be at least 2.0. Shown in the drawing as `[(<current>)]`.
distance L0 L1 <= 5.0         Upper-bound range: gap at most 5.0.
distance L0 L1 2 to 5         Two-sided range: 2 <= gap <= 5.
distance L0 L1 low to high    Two-sided live range: both sides track the params.
distance L0 L1 >= =low        Snapshot: bind the current value of `low` as a literal lower bound.
                              Range syntax also works on `distance <point> <line>`, `distance <endpoint> <endpoint>`, and `distance A0 A1` (concentric arcs). The residual is a one-sided penalty (zero inside the feasible region, linear outside), so an inactive bound contributes no curvature. If a live bound becomes infeasible (e.g. low > high after a param change), the parameter edit is rejected by the existing solver-failure rollback.

Range dimensions are **not counted in the reported DOF**. Because the penalty only fires at/outside the bound, counting it would make the reported DOF flip by one as geometry drags across the boundary (same sketch, same equality constraints, different DOF). The DOF number you see reflects only equality constraints -- the rigid structure of the sketch -- regardless of whether any range is currently active. This applies to `dof`, `dof singular`, and `dof eigenvalues`.
length L0 >= 2                Length range: lower bound.
length L0 <= 10               Length range: upper bound.
length L0 4 to 6              Length range: two-sided.
sweep A0 30 to 270            Sweep range (degrees): two-sided.
angle L0 L1 30 to 60          Angle range: two-sided.
angle L0 L1 >= 90 supplement  Range on the supplementary angle. `closest` / `acute` / `obtuse` modifiers require a single target value and are rejected with a range.
radius A0 >= 2                Radius range: lower bound.
radius A0 2 to 4              Radius range: two-sided.
radius_b EA0 <= 3             Ellipse semi-minor-axis range.
xangle L0 30 to 60            Line-angle range (degrees from x-axis).
hdistance L0.p1 L1.p2 5.0    Horizontal (x-axis) distance between endpoints
hdistance L0.p1 L1.p2 2 to 5 Range on |x|-distance.
vdistance L0.p1 L1.p2 3.0    Vertical (y-axis) distance between endpoints
vdistance L0.p1 L0.p2 >= 3   Range on |y|-distance.
xangle L0 45                  Line angle from x-axis (degrees, CCW positive)
delete d0                     Remove dimension by name
freeze                        Add numeric dimensions for all lines and arcs at current values (line length; arc radius and sweep)
freeze L0 A0                  Freeze specific lines/arcs only
```

### Expression syntax

Dimension values are one of:
- **Numeric literal**: `5`, `3.14` — plain number.
- **Live expression** (the default for anything non-numeric):
  `scale`, `2*scale + 1`, `L0.length` — re-evaluated every solve,
  so the dim tracks whatever the expression refers to. A reference
  to a user parameter, an entity property, or any combination of
  both works. Unresolved names at solve time mark the dim broken.
- **Snapshot**: prefix with `=`. `=2*scale` evaluates `2*scale`
  once, at command time, and stores the result as a literal. Use
  this when you want the current value baked in -- the dim will
  not track further changes.

The `angle` command by default constrains the angle between the line direction vectors (p1 to p2). Two lines always form two angles that sum to 180. Keywords control which angle is constrained:

- **no keyword**: angle between direction vectors (deterministic, depends on p1->p2 direction)
- **`supplement`**: the other angle (180 minus default)
- **`closest`**: auto-select the sector closest to the given value
- **`acute`**: the smaller of the two current angles
- **`obtuse`**: the larger of the two current angles

Negative values are accepted and treated as positive (useful with `angle()` function which may return negative).

If a dimension of the same type already exists on an entity (e.g., calling `radius A0 7` when a radius dimension already exists on A0), the existing dimension is updated in place rather than creating a duplicate.

### Axis Distance (hdistance, vdistance)

`hdistance` constrains the horizontal (x-axis) distance between two endpoints. `vdistance` constrains the vertical (y-axis) distance. Both accept any endpoint ref (P0, L0.p1, L0.p2, A0.center, A0.start, A0.end).

The value is displayed unsigned but applied as a signed constraint — the solver preserves which endpoint is left/right (or above/below) and cannot swap them to satisfy the constraint. This makes the sketch axis-locked (unrotatable), which is appropriate for axis-aligned dimensioning.

```
hdistance L0.p1 L0.p2 4       X-distance between line endpoints
vdistance L0.p1 L1.p2 3       Y-distance between endpoints of different lines
hdistance A0.center L0.p1 2    X-distance from arc center to line endpoint
vdistance P0 L0.p2 1.5 derived  Derived vertical distance
```

### Angle from X-Axis (xangle)

`xangle` constrains the angle of a line or the rotation of an ellipse / elliptic arc, both measured from the positive x-axis in degrees (counter-clockwise positive).

- For a **line** the angle runs from `.p1` toward `.p2`.
- For an **ellipse / earc** (`EA<n>`) the angle is the rotation of the semi-major axis; only ellipses have an optimisable rotation, so `xangle A0 45` on a circular arc is rejected with an error.

```
xangle L0 45                   Line at 45 degrees from x-axis
xangle L0 -30                  Line at -30 degrees (30 degrees below x-axis)
xangle L0 0                    Force line horizontal (alternative to horizontal command)
xangle L0 90                   Force line vertical
xangle L0 derived              Measure current line angle without constraining
xangle EA0 30                  Ellipse major axis at 30 degrees from x-axis
xangle EA0 d0 + 90             Ellipse rotation tracked to the first dim + 90 degrees
xangle EA0 >= -45               Range bound: rotation must stay at or above -45deg
```

In the GUI, the xangle dimension draws a helper x-axis line from the anchor (line `p1` or ellipse center) and an angle arc. The dimension is draggable and editable like other angle dimensions.

### Derived (Reference) Dimensions

Derived dimensions display the measured value but do not constrain the solver. They are shown with parentheses, e.g. `(3.40)`. Add `derived` to any dimension command to create one:

```
length L0 5 derived          Derived length with explicit value
length L0 derived            Derived length (measures current geometry)
radius A0 derived            Derived radius
radius A0 1.5 derived        Derived radius with explicit value
sweep A0 derived             Derived sweep angle
sweep A0 90 derived          Derived sweep with explicit value
angle L0 L1 derived          Derived angle (measures current geometry)
angle L0 L1 45 derived       Derived angle with explicit value
distance L0.p1 L1.p2 derived Derived distance (measures current geometry)
distance L0.p1 L1.p2 5 derived Derived distance with explicit value
```

### Driven (Freeze Current Value)

The `driven` keyword creates a constraining dimension at the current measured value — effectively "freezing" the geometry without having to type the exact value. Unlike `derived`, it reduces DOF.

```
length L0 driven             Constraining length at current measured value
radius A0 driven             Constraining radius at current value
sweep A0 driven              Constraining sweep at current value
angle L0 L1 driven           Constraining angle at current value
distance L0.p1 L1.p2 driven  Constraining distance at current value
hdistance L0.p1 L0.p2 driven Constraining hdistance at current value
vdistance L0.p1 L0.p2 driven Constraining vdistance at current value
xangle L0 driven             Constraining xangle at current value
```

Toggle between derived and driven:

```
set_derived d0               Make dimension derived (removes constraint)
set_driven d0                Make dimension driven (adds constraint, keeps current value)
set_driven d0 5              Make driven with a new value
set_driven d0 "width*2"     Make driven with a parametric expression
```

## Locking

```
lock P0                      Lock point at current position
lock L0.p1                   Lock line endpoint at current position
lock L0.p1 5,3               Lock at specific position
unlock P0                    Unlock point
unlock L0.p1                 Unlock endpoint
```

## Parameters

User-defined named parameters that can be referenced in expressions and dimensions.

```
param width 10               Create parameter — shows: Added width = 10 (10.0000)
param height width * 2       Expression — shows: Added height = width * 2 (20.0000)
param width 15               Update existing — shows: Updated width = 15 (15.0000)
del_param width              Delete parameter
rename_param width w         Rename (propagates to all references)
```

## Style

```
style L0 solid               Set line style
style L0 dashed
style L0 dashdot
style A0 dashed              Also works for arcs
style L0                     Query current style
```

## Selection

`select` adds to the current selection. `select all/chain/linked` replaces it. Use `deselect` to remove entities or clear all.

```
select L0                    Select whole line
select L0.p1                 Select line endpoint
select A0.center             Select arc center
select L0 L1 P0              Select multiple
select all                   Select all entities
select L0 chain              Select all entities connected via coincident endpoints
select L0 linked             Select all entities sharing any constraint, recursively
select L0 sequence           Select the end-to-end sequence through L0, up to an end or a branch
select M0                    Select a meta-constraint (`offset selection ...` then edits it)
deselect                     Clear selection
deselect L0                  Remove specific entity from selection
list selection               Show current selection
```

## Geometric Functions

Functions that return coordinates (usable wherever a coordinate is expected):

| Function | Returns | Description |
|----------|---------|-------------|
| `intersect(L0, L1)` | coord | Intersection of two lines |
| `midpoint(L0)` | coord | Midpoint of a line segment |
| `project(P0, L0)` | coord | Projection of point onto line |
| `along(L0, 0.5)` | coord | Point at fraction along line (0=p1, 1=p2) |
| `arc_point(A0, 45)` | coord | Point on arc at angle (degrees) |
| `rotate(P0, center, 90)` | coord | Rotate point around center |
| `mirror(P0, L0)` | coord | Mirror point across line |
| `tangent(L0)` | coord | Unit tangent vector (p1 to p2 direction) |
| `normal(L0)` | coord | Unit normal vector (perpendicular to tangent) |

Functions that return scalars (usable in expressions):

| Function | Returns | Description |
|----------|---------|-------------|
| `dist(P0, P1)` | scalar | Distance between two points |
| `dist(P0, L0)` | scalar | Perpendicular distance from point to line |
| `angle(L0, L1)` | scalar | Angle between two lines (degrees, signed) |

**Note on `angle()`**: Returns a signed value that may be negative or supplementary. Use `abs(angle(L0, L1))` for a positive value. Feeding `angle()` directly into the `angle` command may fail if the sign/sector doesn't match -- use the "Current angle" value from the rejection error message instead.

### Vector Arithmetic

Coordinate-returning functions support arithmetic:

```
add_point L0.p2 + normal(L0) * 3     Offset from endpoint along normal
add_line midpoint(L0) midpoint(L1)    Line between midpoints
add_point rotate(P0, A0.center, 45)   Rotated point
```

## Entity Name Capture

Every entity creation command (`add_line`, `add_rect`, `add_rect3`, `add_rectcenter`, `add_point`, `add_circle`, `add_circle2`, `add_circle3`, `add_circle2t`, `add_circle3t`, `add_arc`, `add_earc`, `add_earc3`, `add_earc_center`, `add_earc_tangent`, `add_earc_rtangent`, `add_ellipse`, `offset_line`, `fillet`, `chamfer`, `mirror`) automatically sets the `_` variable to the created entity's name (for fillet: the arc; for mirror: the first mirrored copy). Dimension names are not captured -- they are shown in the command output and by `list dims`. Use assignment to capture the entity name with a meaningful name:

```
add_line 0,0 5,0; vertical _; length _ 3       Use _ for the last created entity

base = add_line 0,0 5,0                         Capture entity name in variable
horizontal base
length base 3

side = add_line @0,3                             Chain creation with capture
vertical side
coincident base.p2 side.p1                       Use aliases in endpoint refs
```

Both `name = command` and `let name = command` work. The alias is resolved transparently — anywhere you'd write `L0`, you can write `base` instead.

For multi-segment `add_line`, use comma-separated names to capture each segment:

```
a, b, c = add_line 0,0 @1,0 @0,1 @-1,0    Captures L0->a, L1->b, L2->c
horizontal a
perpendicular a b
```

Multi-name assignment also works for the rectangle commands (four
sides in order) and `mirror` (the mirrored copies in source order):

```
bot, right, top, left = add_rect 0,0 5,3
m1, m2 = mirror L0 L1 about L2
```

## Cursor

The cursor is a visible crosshair at a fixed sketch position. It serves as a reference point for relative coordinates and can be used as a coordinate in commands.

```
cursor                       Show current position and tangent (alias: cursor info)
cursor 5,3                   Set to absolute position
cursor L0.p2                 Set to endpoint
cursor @dx,dy                Move relative to current position
cursor on                    Show (at 0,0 if not set) (alias: show)
cursor off                   Hide cursor (alias: hide)
```

The cursor is automatically set to the last created endpoint:
- `add_line` → cursor at p2
- `add_point` → cursor at point position
- `add_circle` → cursor at center

Append `nocursor` to suppress: `add_line 0,0 5,0 nocursor`

Use `cursor` as a coordinate in any command:
```
cursor 5,0
add_line cursor 5,3          Line from cursor position
add_point cursor              Point at cursor
```

## Dimension Text Position

Control where dimension text is rendered:

```
dim_pos d0 offset 1.5        Set perpendicular offset from dimension line
dim_pos d0 along 0.3         Set position along dimension (-0.5 to 0.5)
dim_pos d0 offset @0.5       Relative offset change
dim_pos d0 along @-0.1       Relative along change
```

`info d0` shows current offset and along values.

## Messages

Print messages to the command history output. Supports Markdown formatting. The command itself is not echoed.

```
msg Hello world
msg **Bold** and *italic* text
msg Line 1\nLine 2           Newlines with \n
msg # Heading
msg - item 1\n- item 2       Lists
```

## Session Variables

Assign intermediate results for reuse within the session:

```
p = L0.p2                    Assign coordinate
d = dist(P0, L1)             Assign scalar
n = normal(L0)               Assign vector
add_line p p + n * d          Use in subsequent commands
print p                       Display value
```

`let name = expr` also works (the `let` keyword is optional).

Session variables are accessible as `name.x`, `name.y` for coordinates.

## Introspection

```
print <expression>           Evaluate and display
print L0.length              Entity property
print dist(P0, L0)           Geometric function
print width * 2 + 1          Expression with parameters
info L0                      Line details + constraints and dimensions referencing L0
info L0.p1                   Endpoint position, lock status, constraints, dimensions
info P0                      Point details + constraints and dimensions
info A0                      Arc details + constraints and dimensions
info A0.center               Arc endpoint position
info d0                      Dimension details (meaning, value, expression, derived status)
info width                   Parameter details (value, expression, broken)
list                         List all entities, dimensions, parameters
list all                     Same as list (explicit)
list lines                   List only lines
list points                  List only points
list arcs                    List only arcs
list dims                    List dimensions: d<n>: <meaning> = value (expression, range, derived, broken shown)
list params                  List only parameters
list constraints             List constraints addressable by their own name (H/V, locks, C<n>); dimensions are under list dims
list metas                   List meta-constraints (offsets): M<n>: what they were made of and what they made
list selection               List the current selection
list constr                  List construction entities
list horizontal              Filter by constraint type (also: vertical, parallel,
                             perpendicular, equal, collinear, tangent, coincident,
                             concentric, midpoint, symmetry, point_on, lock)
list radius                  Filter dimensions by kind (also: angle, length, sweep,
                             distance, hdistance, vdistance, xangle)
measure L0                   Single entity: length, angle, radius, positions
measure L0 L1                Two entities: angle, distance, parallel/perpendicular
measure P0 L0                Point-line: perpendicular distance
measure L0 A0                Line-arc: center distance, gap, tangency
find x,y [radius]            Find entities near a coordinate
dof                          Show degrees of freedom (computed in background, may show "pending")
dof analyze                  Analyze free directions: which entities can move and how
dof eigenvalues              Show Hessian eigenvalue spectrum (one per parameter)
dof singular                 Show Jacobian singular values with their parameter directions
dof jacobian                 Show full Jacobian rows (residual + partial derivatives)
cost                         Show current solver cost
```

### Interpreting `dof singular`

`dof singular` runs SVD on the constraint Jacobian. Each singular triplet
is shown on two lines:

```
  1.322876e8  -0.707 A0.start_angle + 0.707 A0.end_angle
           53% tangent_la:L0,A0, 53% tangent_la:L1,A0, 27% coinc_lp1_ae:L0,A0
```

**Line 1** — singular value σ followed by the right-singular-vector as a
linear combination of parameters. Moving along this direction in parameter
space changes the constraint residuals by σ times the step size.

- **σ = 0**: free direction — a true degree of freedom.
- **Small σ**: weakly constrained, near-free direction. Indicates
  near-symmetries or poor conditioning.
- **Large σ**: strongly constrained direction.

Entries are weights (not current values) summing to unit norm. Only
components above 30% of the max weight are shown.

Common vector patterns:
- `1.0 A0.start_angle` — pure motion of a single parameter
- `0.707 A0.start_angle + 0.707 A0.end_angle` — both parameters shift
  together (arc rotates around its center)
- `-0.707 A0.start_angle + 0.707 A0.end_angle` — parameters shift
  oppositely (arc sweep expands)

**Line 2** — constraints contributing to this direction as percentages.
Sums to 100% across all constraints. Tells you *which constraints care
about this direction*:

- For **small σ**: these constraints barely hold this direction. If
  they're cancelling out or weak at the current scale, that's why the
  direction is near-free.
- For **large σ**: these constraints strongly resist motion. For a
  dome-angle direction you'd expect tangent and coincident constraints
  attached to the dome.

### Interpreting `dof jacobian`

Each row shows one constraint residual with its partial derivatives.
Format: `row N cid=K label  r=<residual> |dr|=<row_norm>  [params]`

- `cid` is the internal constraint ID (unique per constraint instance)
- `label` names the constraint (e.g. `parallel:L3,L0`, `tangent_la:L0,A0`)
- `|dr|` is the Euclidean norm of the partial derivatives

Rows with the same `cid` belong to the same constraint instance (e.g. a
coincident constraint produces 2 rows — one for x, one for y — sharing
one cid).

## Undo / Redo / History

```
undo                         Undo last action group
undo 5                       Undo 5 steps
redo                         Redo next action group
redo 3                       Redo 3 steps
history                      Show action history (by group)
history 10                   Show last 10 groups
goto 5                       Jump to history position 5
goto 0                       Jump to initial state
```

## View Control

```
center                       Fit all entities in view
center L0                    Center view on entity
center 5,3                   Center view on coordinate
zoom +                       Zoom in
zoom -                       Zoom out
zoom 2.0                     Set absolute zoom level
```

## File Operations

```
save path.json               Save sketch to file
load path.json               Load sketch from file
clear                        New empty sketch
```

## Multi-Command

Separate commands with `;`:

```
add_line 0,0 5,0; horizontal L0; length L0 3
```

## Session

```
help                         Command summary
help full                    This full reference
exit                         Exit the editor (alias: quit)
```

`ai` is a placeholder that points at the MCP server; use `--mcp` to
give external agents access.

## Examples

### Rectangle with Dimensions

A fully-constrained, parametric rectangle built from four lines.
Changing `width` or `height` afterwards reshapes the rectangle.

```
param width 5
param height 3
bot   = add_line 0,0 5,0
right = add_line @0,3
top   = add_line @-5,0
left  = add_line top.p2 bot.p1
lock bot.p1 0,0
horizontal bot
horizontal top
vertical left
vertical right
length bot width
length right height
```

Notes:

- `add_line` from an existing line's endpoint
  (`add_line top.p2 bot.p1`) and `add_line @dx,dy` from the
  previous endpoint both auto-emit a coincident constraint, so
  no explicit `coincident` is needed.
- Bare names like `width` in a dimension value are **live**:
  they re-evaluate every solve, so a later `param width 7`
  reshapes the rectangle. Type the number directly (`length bot 5`)
  or use the `=expr` snapshot prefix (`length bot =width`) if you
  want the value baked in at command time instead.
- `add_rect 0,0 width,height hv` produces the same shape in one
  command. The four-line form above is a worked example of how
  the building blocks fit together.

### Parametric Triangle

Isoceles triangle with live base length `b` and height `h`: change
the parameter value and the triangle re-solves.

```
param b 4
param h 3
base  = add_line 0,0 4,0
side1 = add_line base.p2 2,3
side2 = add_line side1.p2 base.p1
lock base.p1 0,0
horizontal base
length base b
hdistance base.p1 side1.p2 b/2
vdistance base.p1 side1.p2 h
```

`add_line` from an existing line endpoint auto-emits a coincident
constraint, so no explicit `coincident` commands are needed. The
bare parameter names in the dimension values are live references:
they re-evaluate on every solve, so `param b 6` afterward
reshapes the triangle.

### Offset Line

The `offset` command (see [Offset](#offset)) holds the copy at the
distance and records the operation:

```
l = add_line 0,0 10,0
m = offset l 2                 M0: parallel line 2 to the left, held there; _0 is the new line
offset M0 3                    move it to 3
```

`offset_line` is the bare copy, unconstrained:

```
l = add_line 0,0 10,0
m = offset_line l 2           Creates parallel line offset by 2
parallel l m                   Constrain them to stay parallel
```

Or manually with vector arithmetic:

```
add_line 0,0 10,0
add_line L0.p1+normal(L0)*2 L0.p2+normal(L0)*2
```

### Fillet

Round a shared corner between two connected lines with a tangent arc
of given radius. The corner coincident is broken, both lines are
trimmed back to the tangent points, and the new arc is pinned to the
trimmed endpoints with coincident constraints. Tangent-line-arc
constraints are added on each side (unless `notangent`), and a
driven radius dimension is added on the arc (unless `noradius`).

```
fillet L1 L2 r               Round the shared corner of L1 and L2
fillet L1.pN r               Round the corner at L1's pN endpoint
fillet L1 L2 r notangent     Skip the tangent constraints
fillet L1 L2 r noradius      Skip the radius dimension
fillet L1 L2 L3 L4 r         Variadic: multiple corners in one command
fillet L1.p2 L3.p1 r         Variadic: multiple endpoints
fillet L1 L2 L3.p1 r         Variadic: mix of forms
```

**Variadic form.** Any number of corners can be given in one
command, with the radius as the last token. Each corner is either
`Lx.pN` (one token) or `Lx Ly` (two tokens). The parser walks left
to right and consumes tokens accordingly. The first corner's radius
dimension carries the typed value or expression; subsequent corners
reference that dim by name so all radii stay equal and track a
single source. Corners that can't be filleted (too short,
collinear, etc.) are reported as `FAILED: <reason>` but don't abort
the other corners; the header shows `Filleted N of M corners`. All
actions run inside a single undo group.

**Requirements.** The two lines must share a corner via a direct
line-line coincident (one of `CoincidentLL11`, `CoincidentLL12`,
`CoincidentLL21`, `CoincidentLL22`). Corners tied through an
intermediate point are rejected. The corner angle must be non-zero
and non-180 degrees (neither overlapping nor collinear). Each line
must be at least `r / tan(theta/2)` long, where `theta` is the angle
between the outgoing directions at the corner.

**Geometry.** Given corner `C`, unit directions `u_a`, `u_b` from `C`
toward the far endpoint of each line, and angle `theta` between them:
the tangent points are at distance `r / tan(theta/2)` along each
line, and the arc center is at distance `r / sin(theta/2)` along the
angle bisector. The arc curves into the interior of the corner.

**Example.**

```
add_rect 0,0 5,3 hv
fillet L0 L1 0.5             Round the (5,0) corner
fillet L1 L2 0.5             Round the (5,3) corner
fillet L2 L3 0.5             Round the (0,3) corner
fillet L3 L0 0.5             Round the (0,0) corner
```

### Chamfer

Bevel a shared corner. The corner coincident is broken, both lines
are trimmed back by distance `d` from the corner, a new "chamfer
line" spans the two trim points, and an anchor point is pinned at
the original corner location via `PointOnLine` on each of the two
input lines. Two distance dimensions record the trim: the primary
holds the typed value (literal or live expression); the secondary is
stored as `expr = <primary-dim-name>` so the two legs stay equal and
track the primary parametrically.

```
chamfer L1 L2 d              Bevel the shared corner of L1 and L2
chamfer L1.pN d              Bevel the corner at L1's pN endpoint
chamfer L1 L2 L3 L4 d        Variadic: multiple corners in one command
```

**Variadic form.** Same parser as `fillet`: any number of corner
specs followed by the distance token. First corner's primary dim
carries the typed value; each corner's secondary dim tracks its
own primary; every subsequent corner's primary dim is `expr = <first
corner's primary dim name>`, so every chamfer leg on the sketch is
equal and driven by a single source.

**Requirements.** Same as fillet: the two lines must share a corner
via a direct `coincident_ll*`, the angle must be non-zero and
non-180, and each line must be longer than `d`.

**Example.**

```
add_rect 0,0 5,3 hv
chamfer L0 L1 0.5            Bevel the (5,0) corner
chamfer L1 L2 0.5            Bevel the (5,3) corner
```

### Split

Cut a line or arc where other curves cross it. The target is deleted
and every piece is a new entity with a new name; all references --
constraints, dimensions, expressions -- transfer onto the pieces:

- Endpoint references (`L0.p1`, `A0.start`, ...) follow the piece
  owning that endpoint.
- Whole-entity constraints (`parallel`, `perpendicular`, H/V flags,
  `concentric`, `equal` radius) are replicated onto every piece.
- Tangencies follow the piece containing the contact point.
- Whole-span measures (`equal` length, host-side `midpoint`,
  `symmetry` operands) have no successor and are dropped, reported.
- A driving `length` dimension becomes a point-to-point `distance`
  between the outer endpoints, keeping its `d<n>` id, name, value and
  placement. An arc `sweep` dimension is dropped.
- Whole-arc dimensions (`radius`, `radius_b`, `xangle`) hold for
  every piece: the first piece keeps the original, every other piece
  gets a copy with a new `d<n>` (same value, expression and
  driven/derived state), reported under `copied`.
- Expression strings are rewritten (`L0.p1.x` -> the piece's,
  `L0.length` -> the sum of the pieces' lengths). An expression whose
  referent has no successor marks its dimension/parameter broken.

The cut endpoints are joined with a coincident constraint and pinned
onto the cutter with a point-on constraint (skip with `nopin`).
Adjacent arc pieces get a `concentric` tie. All of it is one undo
group, and the output lists every added, moved, copied and dropped id.

```
split L0 4,0                 Cut at the intersections bracketing (4,0)
split L0 4,1 0.5             Same, but error if (4,1) is farther than 0.5 from L0
split L0 by L1               Cut at every intersection with L1
split A0 by L1 L2            Cut the circle at every crossing with L1 and L2
split L0 4,0 nopin           Do not pin the cut endpoints onto the cutter
```

The coordinate does not have to sit on the curve: it is projected
onto the target and the nearest point wins. The optional radius
bounds that search. A closed circle/ellipse needs at least two
crossings.

All pieces are captured (`_0`, `_1`, ...), so a named-piece trim
composes from split + delete:

```
# split a line where L1 crosses it, then drop the far half
a, b = split L0 by L1
delete b

# same thing directly
trim L0 by L1 forward
```

### Trim

A split whose clicked/named span is deleted instead of kept. Same
reference transfer, same output. With no intersections at all, trim
deletes the whole entity.

```
trim L0 4,0                  Delete the span of L0 containing (4,0)
trim L0 4,1 0.5              Same, with a bounded coordinate search
trim L0 by L1 L2             Delete the span between L1's and L2's crossings
trim L0 by L1 forward        Delete from L1's crossing toward p2/end
trim L0 by L1 backward       Delete from L1's crossing toward p1/start
```

`forward` / `backward` follow the entity's own direction (p1 -> p2,
start -> end); with a twice-crossing cutter, `forward` cuts at the
crossing nearest the end. `trim by` between two cutters removes
everything between their crossings, including any other crossings in
between. On a closed circle/ellipse the `by` forms are rejected (the
boundaries name two complementary spans); use the coordinate form.

### Scale

Uniformly scale lines, arcs and points about a center point. Every
position moves to `center + factor * (position - center)`; radii
multiply by the factor; angles and sweeps do not change. Lock targets
follow their points.

```
scale L0 L1 about P0 2       Double L0 and L1 away from P0
scale A0 about 0,0 0.5       Halve the circle about the origin
scale L0 about L1.p1 1.5     Center named by any endpoint reference
scale selection about P0 w/3 Factor is an expression
scale P1 P2 about P0 2       Points can be scaled too
```

The factor must be positive; `1` is a no-op geometrically. Driving
linear dimensions (`length`, `distance`, `hdistance`, `vdistance`,
`radius`, `radius_b`) whose endpoints all lie inside the scaled set
scale with the geometry: the value multiplies by the factor, literal
range bounds too. Derived dimensions just re-measure. Dimensions that
cannot follow are left untouched and reported with a reason:

- expression-driven values (the expression still means what it says),
- live (non-literal) range bounds,
- linear dimensions spanning unscaled geometry.

The command output lists both groups (`dims scaled:` / `dims
left:`). The whole scale is one undo step.

## Solver Parameters

The solver uses soft Heaviside penalties to prevent degenerate geometry. The `min_length` parameter (default `0.0001`) sets the minimum threshold for:

- **Line length**: lines shorter than `min_length` are penalized to prevent collapse to zero length (which makes direction undefined and breaks tangent/angle constraints).
- **Arc radius**: arcs with radius below `min_length` are penalized.
- **Tangent projection**: when a line-arc tangent constraint has a shared endpoint, the line's free end must project at least `min_length` along the arc tangent direction.

This value is saved with the sketch file. It should not normally need adjustment.

## Best Practices for Parametric Sketches

### Plan the constraint strategy before drawing

Decide on axes of symmetry, which dimensions should be parametric, and your target DOF. DOF 0 means fully constrained (typical for manufacturing drawings). DOF 3 means the shape can translate and rotate freely (good for reusable components).

### Build a skeleton of construction lines first

Before drawing profile geometry, create a frame of construction reference lines:

1. **Centerline** along the axis of symmetry (or the main axis for asymmetric parts)
2. **Datum lines** perpendicular to the centerline at key positions (end of part, feature locations, section boundaries)
3. **Dimension the skeleton**: spacing between datum lines, distances from centerline
4. **Then build profile geometry** constrained to these references (coincident, distance, symmetric)

Dimension profile features from the nearest datum line, not from other profile edges. This makes the constraint graph hub-and-spoke (all features reference datums) instead of a fragile chain (each feature depends on the previous one). Changing one feature only requires editing its distance from its datum — nothing else shifts.

This applies to both symmetric and asymmetric parts. Even a simple bracket benefits from a horizontal and vertical reference line that everything dimensions against.

### Prefer geometric constraints, then add dimensions

Apply constraints in this order:
1. **Topology**: `coincident`, `point_on`, `tangent` — what connects to what
2. **Orientation**: `parallel`, `perpendicular`, `symmetric` — relative directions
3. **Size**: `length`, `radius`, `distance`, `angle` — magnitudes

Prefer `perpendicular L0 L1` over `angle L0 L1 90` — geometric constraints are exact, more robust, and don't clutter the dimension list.

### Make parameter changes incrementally

Large parameter jumps (e.g., changing a scale variable from 1 to 10) can cause solver convergence issues, potentially losing constraints or producing unexpected geometry. Use incremental changes for robust behavior (e.g., 1 -> 2 -> 5 -> 10). This is especially important for parameters that affect many constraints simultaneously.

### Use construction lines as reference geometry

Add construction lines with `style L0 dashdot` to define axes of symmetry and reference frames. These constrain like normal lines but are visually distinct. Anchor both endpoints to known geometry (e.g., `coincident L0.p1 A0.center` + `midpoint P0 L0`) rather than using length + angle, which adds unnecessary DOF complexity.

### Use symmetry instead of duplicating constraints

For symmetric shapes, fully constrain one side and use `symmetry P0 L_axis P1` to mirror points. This is cleaner than constraining both sides independently. Symmetry makes some constraints redundant:

- Two points symmetric about a line means the segment between them is **automatically perpendicular** to that line — don't add `perpendicular`.
- If endpoints of two lines are mirrored, their lengths are **automatically equal** — don't add `equal`.
- If two points are symmetric about a line and both lie on an arc centered on that line, tangent at those points is **already implied** — don't add `tangent`.
- If a line endpoint is coincident with an arc center, `point_on` for that center on the line is **redundant**.
- If a line is parallel to the symmetry axis and its mirror exists, `equal` between the line and its mirror is **redundant**.

### Prefer relative constraints over absolute ones

- Use `perpendicular L0 L1` instead of `horizontal` / `vertical` -- absolute constraints pin geometry to the global frame and prevent free rotation.
- Use `perpendicular L0 L1` instead of `angle L0 L1 90` -- the perpendicular constraint is more efficient, expresses intent more clearly, and doesn't create a dimension entry (keeps `list dims` cleaner).
- Use `angle`, `distance`, `symmetry` between entities rather than `lock` for positioning.
- Only use `horizontal` / `vertical` / `lock` when you intentionally want to fix something to the global coordinate system.
- For a DOF=0 sketch, ideally use at most one `lock` constraint (to pin position) and one `horizontal`/`vertical` (to pin orientation). All other constraints should be relative.

### Monitor DOF as you build

Check `dof` after each batch of constraints. If DOF doesn't drop as expected, diagnose immediately rather than building on a broken foundation. Use `dof analyze` to see which directions remain free, `list constraints` to verify what's applied.

### Use named parameters, not magic numbers

Define `param width 10; param height 5` and write dimensions as expressions: `length L0 width`, `length L1 height`. When the same value appears in multiple dimensions, a named parameter ensures they stay in sync and makes the sketch self-documenting. Use `param scale 1` with `radius A0 5*scale` for globally scalable sketches.

### Constrain one side, then close the loop

For closed shapes: fully constrain one chain of segments (lengths + angles between adjacent segments), then use symmetry or explicit constraints to close the loop. Be aware that closure conditions can make some constraints redundant.

### Use point_on for positioning features

`point_on A0.center L0` constrains an arc's center to lie on a line (removes 1 DOF). Use this for placing features like circles on a construction line.

### Use execute_script for batching

Send multiple commands at once using `execute_script`. Use `#` comments to document sections. Check results with `list` after major operations (or `get_sketch_state` MCP tool).

### Coordinate system

Y-axis points up (math convention, not screen convention). Plan your geometry accordingly -- positive Y is upward.

### Undo recovers mistakes

`undo` reverses the last operation, including grouped operations (e.g., a constraint that created helper points undoes as one unit). Use `undo` freely when experimenting with constraint strategies.

### DOF check catches redundant constraints

When you add a constraint, the solver checks that DOF actually decreases. If it doesn't, the constraint is rejected -- it is either redundant (already implied by existing constraints) or the geometry already satisfies it trivially. Check `dof analyze` to see which directions remain free and reconsider your constraint strategy.

### DOF=3 pattern for reusable components

To create a freely movable/rotatable component (DOF=3), use only relative constraints — no `horizontal`, `vertical`, or `lock`:

```
# Example: rhombus, DOF=3 (translate + rotate)
# Four sides
s0 = add_line -3,0 0,2
s1 = add_line 0,2 3,0
s2 = add_line 3,0 0,-2
s3 = add_line 0,-2 -3,0
equal s0 s1
equal s1 s2
equal s2 s3
# Construction diagonals through opposite vertices
d_h = add_line s0.p1 s2.p1
d_v = add_line s1.p1 s3.p1
style d_h dashdot
style d_v dashdot
# Shape is defined by diagonal lengths
length d_h 6
length d_v 4
# DOF=3: shape can translate and rotate freely
```

Key pattern: use variable names (`s0`, `top`, `left`) instead of absolute entity names (`L0`, `A0`). Constrain with `equal`, `length`, `perpendicular` between entities. Avoid `horizontal`, `vertical`, and `lock` entirely.

**Converting DOF=0 to DOF=3**: To make an existing fully-constrained drawing into a reusable component:
1. Remove all `lock` constraints (`list lock` to find them)
2. Remove `horizontal` / `vertical` constraints — replace with `parallel` or `perpendicular` to a construction reference line
3. Remove absolute `distance` constraints — replace with relative distances between entities
4. The remaining DOF=3 allows free translation (X, Y) and rotation

## Common Shape Recipes

### Semicircle (DOF=0)

```
a = add_arc -5,0 5,0 0,5
sweep a 180                     # -1 DOF
radius a 5                     # -1 DOF
# DOF=3: semicircle shape is fixed, can translate + rotate
lock a.center 0,0              # -2 DOF
# Construction line connecting endpoints pins orientation
base = add_line a.start a.end  # auto-connects to arc endpoints: -4 DOF
style base dashdot
horizontal base                # -1 DOF -> DOF=0
```

### Equilateral Triangle (DOF=0)

```
# 3 lines, auto-connected: 12 - 6 = 6 DOF
s0 = add_line 0,0 5,0
s1 = add_line 5,0 2.5,4.33
s2 = add_line 2.5,4.33 0,0
equal s0 s1                    # -1 DOF
equal s1 s2                    # -1 DOF
# With all sides equal, all angles are 60 -- no angle constraint needed
length s0 5                    # -1 DOF -> DOF=3: shape fixed, can translate + rotate
lock s0.p1 0,0                 # -2 DOF
horizontal s0                  # -1 DOF -> DOF=0
```

For a general (non-equilateral) triangle, replace `equal` constraints with `length` for each side, or use 2 lengths + 1 angle.

### Rectangle (DOF=0)

```
# 4 lines using relative coordinates, auto-connected: 16 - 8 = 8 DOF
s0 = add_line 0,0 @10,0
s1 = add_line @0,5
s2 = add_line @-10,0
s3 = add_line @0,-5
perpendicular s0 s1            # -1 DOF
perpendicular s1 s2            # -1 DOF
perpendicular s2 s3            # -1 DOF
length s0 10                   # -1 DOF (width)
length s1 5                    # -1 DOF (height) -> DOF=3: shape fixed, can translate + rotate
lock s0.p1 0,0                 # -2 DOF
horizontal s0                  # -1 DOF -> DOF=0
```

### Slot / Oblong (DOF=0)

```
# Two lines (offset bot so tangent constraints are not degenerate)
top = add_line 0,2 10,2
bot = add_line 11,-3 1,-3
# Arcs connecting line endpoints
right = add_arc top.p2 bot.p1 13,0
left = add_arc bot.p2 top.p1 -2,0
# Both lines tangent to both arcs (implies parallel + equal length)
tangent top right
tangent bot right
tangent top left
tangent bot left
equal right left
# Size
length top 10
radius right 2
# DOF=3: slot shape fixed, can translate + rotate
lock top.p1 0,2
horizontal top                 # DOF=0
```

### Regular Polygon (DOF=3, add lock+horizontal for DOF=0)

For an N-sided regular polygon, use N lines with coincident endpoints (auto-connected), all equal length, and (N-3) interior angle constraints. Set the first angle numerically and capture its dimension name with `ia = angle ...`, then reference it for the rest — this creates expression dimensions that track the first angle's value. Interior angle = (N-2)*180/N degrees. Example for a hexagon (N=6, angle=120):

```
# 6 lines, auto-connected: 24 - 12 = 12 DOF
s0 = add_line 5,0 2.5,4.33
s1 = add_line 2.5,4.33 -2.5,4.33
s2 = add_line -2.5,4.33 -5,0
s3 = add_line -5,0 -2.5,-4.33
s4 = add_line -2.5,-4.33 2.5,-4.33
s5 = add_line 2.5,-4.33 5,0
# All sides equal: -5 DOF
equal s0 s1; equal s1 s2; equal s2 s3; equal s3 s4; equal s4 s5
# Interior angles: first sets the value, rest reference it: -3 DOF
ia = angle s0 s1 120 closest
angle s1 s2 ia closest; angle s2 s3 ia closest
length s0 5                    # -1 DOF -> DOF=3: shape fixed, can translate + rotate
# Fix position and orientation for DOF=0:
lock s0.p1 5,0                 # -2 DOF
horizontal s1                  # -1 DOF -> DOF=0
```
