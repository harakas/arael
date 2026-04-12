# Random Ideas

Ideas to consider for future development. Not committed to, just collected.

## add_datum shortcut

Creating a datum line is verbose: create line, style, perpendicular, point_on, distance, length. A shortcut like `add_datum cl 8.0` would create a construction line perpendicular to `cl` at station 8.0, fully constrained. Would make the skeleton-first workflow practical.

## Horizontal/vertical distance dimensions

`hdistance P0 P1 5` — constrain only the X-component distance between two points. `vdistance P0 P1 3` — constrain only the Y-component. Different philosophy from perpendicular distance: these make the sketch axis-locked (unrotatable), which is fine for many practical cases. Would simplify datum-based dimensioning for axis-aligned sketches.

## DOF echo after every command

Print DOF after every modifying command in the command interface. Helps agents and users track constraint progress without manually typing `dof`. Needs a toggle (`set dof_echo on/off`) since it's expensive for large sketches. Default: on for interactive, off for batch scripts.

## Worked skeleton example in docs

A step-by-step example showing: create centerline, create datum lines, position profile vertices from datums, use symmetry. Currently the best practices describe the philosophy but the examples use chain constraints. A skeleton workflow example would be the most impactful doc addition for agents.
