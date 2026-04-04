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
| `symmetry` (point-point about line) | 2 |
| `symmetry` (line-line about line) | 4 |
| `lock` (point) | 2 |
| `lock` (line endpoint) | 2 |
| Dimension (`length`, `radius`, `sweep`, `angle`, `distance`) | 1 |

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
angle s0 s1 60

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
add_line x2,y2               Continue from last endpoint (chaining)
add_line @dx,dy              Continue with relative offset
add_point x,y [nocursor]     Create a free point
add_circle cx,cy radius      Create a circle (full arc)
add_arc x1,y1 x2,y2 xm,ym  Create an arc through start, end, midpoint
offset_line L0 distance      Create a parallel line offset by distance (alias: offset)
```

### Auto-Coincident

When creating geometry (`add_line`, `add_circle`, `add_arc`), endpoints that are within 1e-3 distance of existing endpoints are automatically connected with coincident constraints. Lines snap to line endpoints, points, and arc endpoints. Circles and arcs snap their center (and start/end for arcs) to line endpoints, other arc endpoints, and points.

```
add_line 0,0 5,0              L0
add_line 5,0 5,3              L1 — auto-connects L1.p1 to L0.p2
add_circle 5,0 1              A0 — auto-connects A0.center to L0.p2
add_circle 0,0 2              A1 — auto-connects to L0.p1
add_line 0,0 5,3              L2 — auto-connects L2.p2 to A0.center
```

The output shows what was connected: `Added L1: ... [connected: L1.p1=L0.p2]`

Append `noconnect` to suppress auto-connection:
```
add_line 5,0 5,3 noconnect      No auto-coincident
add_circle 5,0 1 noconnect      No auto-coincident
add_arc 0,0 5,0 2,3 noconnect   No auto-coincident
```

### Line Chaining

When `add_line` is given only one coordinate, it starts from the last created endpoint:

```
add_line 0,0 5,0             L0: (0,0)-(5,0)
add_line @0,3                L1: (5,0)-(5,3)   continues from L0.p2
add_line @-5,0               L2: (5,3)-(0,3)
add_line L0.p1               L3: (0,3)-(0,0)   close the rectangle
```

## Deletion

```
delete L0                    Delete a line
delete P0                    Delete a point
delete A0                    Delete an arc
```

## Constraints

```
horizontal L0 [L1 ...]      Make lines horizontal
vertical L0 [L1 ...]        Make lines vertical
parallel L0 L1               Make two lines parallel
perpendicular L0 L1          Make two lines perpendicular (alias: perp)
equal L0 L1                  Equal length (lines) or equal radius (arcs)
collinear L0 L1              Make two lines collinear
tangent L0 A0                Tangent: line-arc or arc-arc
coincident L0.p2 L1.p1       Coincident: any endpoint pair
concentric A0 A1             Concentric arcs
midpoint P0 L0               Point at midpoint of line
midpoint L0.p1 L1            Line endpoint at midpoint of another line
midpoint P0 A0               Point at angular midpoint of arc
midpoint L0.p1 A0            Line endpoint at angular midpoint of arc
symmetry L0 L1 L2            Lines L0,L2 symmetric about L1
symmetry P0 L0 P1            Points P0,P1 symmetric about L0
symmetry L0.p1 L1 L2.p1     Endpoints symmetric about L1
point_on P0 L0               Point on line
point_on L0.p1 A0            Line endpoint on arc
point_on A0.center L0        Arc center on line (creates helper point internally)
```

All constraint and dimension commands check that DOF decreases after application. If a constraint doesn't reduce DOF (redundant or degenerate), it is rejected. Append `force` to skip this check:

```
equal L0 L1 force            Skip DOF check (use in scripts for known-valid constraints)
length L0 5 force            Skip DOF check for dimensions
```

### Removing Constraints

```
remove_constraint L0 horizontal
remove_constraint L0 L1 parallel
remove_constraint L0 L1 perpendicular
remove_constraint L0 L1 equal
remove_constraint A0 A1 equal_radius
remove_constraint L0 L1 collinear
remove_constraint L0 A0 tangent
remove_constraint A0 A1 concentric
remove_constraint L0.p2 L1.p1 coincident
remove_constraint P0 L0 point_on
remove_constraint L0.p1 A0 point_on
remove_constraint A0.center L0 point_on
remove_constraint P0 L0 P1 symmetry
remove_constraint L0 L1 L2 symmetry
remove_constraint P0 L0 midpoint
remove_constraint L0.p1 L1 midpoint
remove_constraint L0.p1 lock
```

Alias: `rc` (e.g., `rc L0 horizontal`)

## Dimensions

```
length L0 5                  Set line length (numeric)
length L0 L0.length          Evaluate expression, use as numeric value
length L0 =2*scale           Live expression (tracks parameter changes)
length L0 {2*scale}          Live expression (alternative syntax)
radius A0 1.5                Set arc radius
sweep A0 180                 Set arc sweep angle (degrees)
angle L0 L1 45               Set angle between lines (degrees, auto-selects sector)
distance L0.p1 L1.p2 5.0     Point-point distance
distance P0 L0 3.0            Point-line distance
remove_dim d0                 Remove dimension by name
freeze                        Add numeric dimensions for all entities at current values
freeze L0 A0                  Freeze specific entities only
```

### Expression syntax

Dimension values can be:
- **Numeric**: `5`, `3.14` — constant value
- **Evaluate-once**: `L0.length`, `2*scale` — expression evaluated to a number at command time
- **Live expression**: `=2*scale` or `{2*scale}` — re-evaluated when parameters change. Prefix with `=` or wrap in `{}`.

The `angle` command automatically selects the sector (direct or supplementary) that is closest to the given value. For example, if two lines form a 30-degree acute angle, `angle L0 L1 45` targets the direct sector, while `angle L0 L1 150` targets the supplementary sector.

If a dimension of the same type already exists on an entity (e.g., calling `radius A0 7` when a radius dimension already exists on A0), the existing dimension is updated in place rather than creating a duplicate.

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
| `angle(L0, L1)` | scalar | Angle between two lines (degrees) |

### Vector Arithmetic

Coordinate-returning functions support arithmetic:

```
add_point L0.p2 + normal(L0) * 3     Offset from endpoint along normal
add_line midpoint(L0) midpoint(L1)    Line between midpoints
add_point rotate(P0, A0.center, 45)   Rotated point
```

## Entity Name Capture

Entity creation commands (`add_line`, `add_point`, `add_circle`, `add_arc`, `offset_line`) automatically set the `_` variable to the created entity's name. Use assignment to capture it with a meaningful name:

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

## Cursor

The cursor is a visible crosshair at a fixed sketch position. It serves as a reference point for relative coordinates and can be used as a coordinate in commands.

```
cursor                       Show current position
cursor 5,3                   Set to absolute position
cursor L0.p2                 Set to endpoint
cursor @dx,dy                Move relative to current position
cursor on                    Show (at 0,0 if not set)
cursor off                   Hide cursor
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
info L0                      Line details + all constraints referencing L0
info L0.p1                   Endpoint position, lock status, constraints
info P0                      Point details + constraints
info A0                      Arc details + constraints
info A0.center               Arc endpoint position
info d0                      Dimension details (value, expression, derived status)
info width                   Parameter details (value, expression, broken)
list                         List all entities, dimensions, parameters
list all                     Same as list (explicit)
list lines                   List only lines
list points                  List only points
list arcs                    List only arcs
list dims                    List only dimensions (shows "derived" tag)
list params                  List only parameters
list constraints             List all active constraints
find x,y [radius]            Find entities near a coordinate
dof                          Show degrees of freedom (computed in background, may show "pending")
dof analyze                  Analyze free directions: which entities can move and how
cost                         Show current solver cost
```

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

## Examples

### Rectangle with Dimensions

```
bot = add_line 0,0 5,0
right = add_line @0,3
top = add_line @-5,0
left = add_line bot.p1
coincident bot.p2 right.p1
coincident right.p2 top.p1
coincident top.p2 left.p1
coincident left.p2 bot.p1
horizontal bot; horizontal top
vertical left; vertical right
lock bot.p1 0,0
param width 5; param height 3
length bot "width"; length right "height"
```

### Parametric Triangle

```
param b 4; param h 3
base = add_line 0,0 b,0
side1 = add_line base.p2 b/2,h
side2 = add_line side1.p2 base.p1
coincident base.p2 side1.p1
coincident side1.p2 side2.p1
coincident side2.p2 base.p1
lock base.p1 0,0
```

### Offset Line

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

## Best Practices for Parametric Sketches

### Plan the constraint strategy before drawing

Decide on axes of symmetry, which dimensions should be parametric, and your target DOF. DOF 0 means fully constrained (typical for manufacturing drawings). DOF 3 means the shape can translate and rotate freely (good for reusable components).

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

Check `dof` after each batch of constraints. If DOF doesn't drop as expected, a constraint is redundant. If the solver rejects a constraint, it conflicts with the current system. Use `list constraints` and `get_sketch_state` to verify the current state.

### Use parameters for scalable dimensions

Define `param scale 1` (or `param width 10`, `param height 5`, etc.) and write dimensions as expressions: `radius A0 5*scale`, `length L0 width`. This makes the sketch parametric. Angles generally don't need parameters since they are scale-independent.

### Constrain one side, then close the loop

For closed shapes: fully constrain one chain of segments (lengths + angles between adjacent segments), then use symmetry or explicit constraints to close the loop. Be aware that closure conditions can make some constraints redundant.

### Use point_on for positioning features

`point_on A0.center L0` constrains an arc's center to lie on a line (removes 1 DOF). Use this for placing features like circles on a construction line.

### Use execute_script for batching

Send multiple commands at once using `execute_script`. Use `#` comments to document sections. Check results with `get_sketch_state` after major operations.

### Coordinate system

Y-axis points up (math convention, not screen convention). Plan your geometry accordingly -- positive Y is upward.

### Undo recovers mistakes

`undo` reverses the last operation, including grouped operations (e.g., a constraint that created helper points undoes as one unit). Use `undo` freely when experimenting with constraint strategies.

### Watch out for degenerate constraints

Some constraints are accepted by the solver but don't actually reduce DOF because their Jacobian is zero at the current geometry. This happens when:

- `tangent` between a line and arc at a point where the line is already radial (perpendicular to the arc)
- `distance` set to the maximum possible value (e.g., chord equal to diameter)
- Any constraint that is algebraically satisfied by the current configuration but has zero gradient

The solver gives no warning — the constraint appears in `list constraints` but `dof` doesn't change. If a constraint doesn't reduce DOF as expected, try an algebraically equivalent but non-degenerate formulation. For example, use `distance A0.center L1.p2` (center-to-endpoint) instead of `tangent` at the diameter.

Use `dof analyze` to see exactly which directions remain free.

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
ia = angle s0 s1 120
angle s1 s2 {ia}; angle s2 s3 {ia}
length s0 5                    # -1 DOF -> DOF=3: shape fixed, can translate + rotate
# Fix position and orientation for DOF=0:
lock s0.p1 5,0                 # -2 DOF
horizontal s1                  # -1 DOF -> DOF=0
```
