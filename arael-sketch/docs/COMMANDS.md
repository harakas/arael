# Arael Sketch — Command Reference

Arael Sketch is a 2D parametric constraint-based sketch editor. You build geometry (lines, arcs, circles, points), apply geometric constraints (horizontal, parallel, coincident, tangent, etc.), and add dimensions to control sizes and distances. The solver automatically adjusts geometry to satisfy all constraints and dimensions in real time.

This is the complete command reference. Commands can be entered in the command panel (press `/` to open, `Escape` to close) or sent programmatically via the MCP server. Commands can be chained with `;`.

## Coordinate System

The coordinate system uses standard math convention: **X-axis points right, Y-axis points up**. Positive angles are counter-clockwise. This is NOT screen convention (where Y points down).

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

- **Entity properties**: `L0.p1.x`, `L0.p2.y`, `L0.length`, `L0.angle`, `A0.radius`, `A0.center.x`, `P0.x`, `P0.y`
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
symmetry L0 L1 L2            Lines L0,L2 symmetric about L1
symmetry P0 L0 P1            Points P0,P1 symmetric about L0
symmetry L0.p1 L1 L2.p1     Endpoints symmetric about L1
point_on P0 L0               Point on line
point_on L0.p1 A0            Line endpoint on arc
```

### Removing Constraints

```
remove_constraint L0 horizontal
remove_constraint L0 L1 parallel
remove_constraint L0 L1 perpendicular
remove_constraint L0 L1 equal
remove_constraint L0 L1 collinear
remove_constraint L0 A0 tangent
remove_constraint A0 A1 concentric
remove_constraint L0.p1 lock
```

Alias: `rc` (e.g., `rc L0 horizontal`)

## Dimensions

```
length L0 5.0                Set line length (numeric)
length L0 "width * 2"        Set line length (expression)
radius A0 1.5                Set arc radius
angle L0 L1 45               Set angle between lines (degrees, auto-selects sector)
distance L0.p1 L1.p2 5.0     Point-point distance
distance P0 L0 3.0            Point-line distance
remove_dim d0                 Remove dimension by name
```

The `angle` command automatically selects the sector (direct or supplementary) that is closest to the given value. For example, if two lines form a 30-degree acute angle, `angle L0 L1 45` targets the direct sector, while `angle L0 L1 150` targets the supplementary sector.

### Derived (Reference) Dimensions

Derived dimensions display the measured value but do not constrain the solver. They are shown with parentheses, e.g. `(3.40)`. Add `derived` to any dimension command to create one:

```
length L0 5 derived          Derived length with explicit value
length L0 derived            Derived length (measures current geometry)
radius A0 derived            Derived radius
radius A0 1.5 derived        Derived radius with explicit value
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

```
select L0                    Select whole line
select L0.p1                 Select line endpoint
select A0.center             Select arc center
select L0 L1 P0              Select multiple
deselect                     Clear selection
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
