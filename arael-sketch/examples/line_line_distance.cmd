# Line-to-line distance dimension demo.
#
# Create two nearly-parallel lines and set a perpendicular distance
# between them. The `distance L0 L1 <val>` command implicitly applies
# a Parallel constraint between the two lines before adding the
# LineLineDistance dimension, so the lines snap exactly parallel at
# the requested gap.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/line_line_distance.cmd
#
# Expected: list shows both `parallel L0 L1` and a `distance L1.p1 L0`
# entry, and the lines are exactly parallel with a 5-unit perpendicular
# gap.

add_line 0,0 4,0
add_line 0,2 4,3
distance L0 L1 5.0
info L0
info L1
list
