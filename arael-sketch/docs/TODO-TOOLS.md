# Drawing Tools and Aids — TODO

Brainstorm of tools to add to the command interface and/or GUI. Organized by priority and category. Each tool should work as both a command and a GUI action.

## Shape Drawing Tools

### Rectangles
- **Corner rectangle** — Two opposite corners. Creates 4 lines + 4 coincident + 2 horizontal + 2 vertical constraints. The most common shape tool.
- **Center rectangle** — Center point + corner. Same constraints, centered placement.
- **3-point rectangle** — First two clicks define one edge (sets angle), third click defines width. Creates a rotated rectangle with appropriate angle constraints instead of H/V.

### Slots
- **Straight slot** — Two center points + radius. Creates 2 semicircular arcs + 2 parallel lines with tangent + coincident constraints. Very common in mechanical design for bolt adjustment.
- **Center-point slot** — Center of slot + one end + width. Easier to place symmetrically.
- **Arc slot** — Slot following a circular arc path. Defined by arc center, radius, sweep, and slot width. For rotary adjustment features.

### Polygons
- **Regular polygon (inscribed)** — Center + vertex + N sides. Creates N equal-length lines with equal angles. Our `polygon128.cmd` recipe already does this via scripting — a proper tool would be cleaner.
- **Regular polygon (circumscribed)** — Center + edge midpoint + N sides. Edges tangent to a construction circle.
- **Edge-defined polygon** — First two clicks define one edge, polygon built from that.

## Circle and Arc Construction

### Circles
- **3-point circle** — Three points on circumference define a unique circle. Useful when the circle must pass through specific geometry.
- **2-point circle (diameter)** — Two diametrically opposite points. Center is midpoint.
- **Circle tangent to 2 lines + radius** — Creates a circle tangent to two selected lines with a given radius. Like a fillet that stays as a full circle.
- **Circle tangent to 3 entities** — Apollonius problem: circle tangent to three lines/circles/arcs. Powerful for complex tangent constructions.

### Arcs
- **3-point arc** — Start, end, point on arc. Most flexible — no need to know center. We have circumscribed arc creation but not a direct 3-point command.
- **Tangent arc** — Arc starting tangent to the endpoint of the last drawn line/arc. For smooth G1-continuous profiles.
- **Start + end + tangent direction arc** — Arc from start to end, tangent to a line at the start point.

## Modification Tools

### Fillet and Chamfer
- **Fillet** — Round intersection of two lines/arcs with a tangent arc of given radius. Trims original geometry to tangent points. Adds arc + tangent constraints automatically. The most-requested modification tool.
- **Chamfer** — Replace a corner with a straight cut. Equal-distance (45 degree) or two-distance (asymmetric). Adds a line + coincident constraints.

### Trim, Extend, Split
- **Trim** — Click on a segment between two intersections to delete it. Essential for cleaning construction geometry into final profiles.
- **Power/drag trim** — Drag across multiple segments to trim them all. Huge time saver.
- **Extend** — Lengthen a line or arc until it intersects another entity.
- **Split** — Break a segment at a point into two segments sharing that point. Needed when constraints must apply to part of an entity.

### Offset
- **Offset curve** — Copy of a line/arc/chain at uniform distance. Line offset = parallel line. Circle offset = concentric circle. Profile chain offset = offset profile with fillets at convex corners. Essential for wall thicknesses, clearances.

### Mirror
- **Mirror entities** — Select geometry + mirror axis (construction line). Create mirrored copies with symmetry constraints. The mirrored geometry stays linked. We have `symmetry` constraint but no batch mirror tool that duplicates geometry.

### Pattern / Array
- **Linear pattern** — Replicate selected geometry along a direction with count + spacing. Implemented as copied entities with equal-spacing constraints.
- **Circular pattern** — Replicate around a center point with count + angular spacing. For bolt-hole patterns, gear teeth.

## Construction Aids

### Tangent Constructions
- **Tangent line from point to circle** — Line from external point tangent to circle (2 solutions, pick by cursor).
- **Common tangent to two circles** — Line tangent to two circles (4 solutions: 2 external, 2 internal).

### Geometric Construction
- **Perpendicular bisector** — Construction line through midpoint of a segment, perpendicular to it.
- **Angle bisector** — Construction line bisecting the angle between two lines.
- **Intersection point** — Create point at the intersection of two entities (even if they don't physically touch within drawn extent).

## Constraint Inference (GUI)

Auto-detect and apply constraints during drawing based on cursor proximity:
- **Horizontal/vertical** — Snap line to H/V when within tolerance of axis-aligned.
- **Tangent** — When starting/ending arc at existing circle/arc, infer tangent.
- **Perpendicular** — When drawing line near 90 degrees to existing line.
- **Equal length** — When new line is drawn close to same length as adjacent line.
- **Midpoint** — Snap to midpoint of existing line.

We already have coincident inference (auto-connect). The others would reduce manual constraint work.

## Lower Priority

- **Ellipse / elliptical arc** — Full or partial ellipse. Useful but complex constraint formulation (5 params: center, semi-major/minor, rotation).
- **B-spline** — Control-point or interpolation spline. High complexity for constraint solver (each control point adds 2 DOF). Needed for free-form profiles.
- **Construction mode toggle** — Convert entities between real/construction. We have `style` command but no one-click toggle in GUI.
- **Text as geometry** — Place text as line/arc entities for engraving. Heavy on entities.
- **DXF/SVG import** — Import 2D geometry into sketch. Would need constraint cleanup.
