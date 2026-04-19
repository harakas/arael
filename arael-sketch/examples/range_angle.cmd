# Angle range demo.
#
# Two lines at 45 degrees, then constrain the angle to [10, 30]. The
# upper bound activates, rotating the second line so the angle lands
# at 30.
#
# `closest`, `acute`, and `obtuse` are value-selection heuristics
# that require a single target value; they are rejected with a
# range. `supplement` IS allowed and picks which of the two angles
# (theta vs 180-theta) the range applies to.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_angle.cmd

l0 = add_line 0,0 5,0
l1 = add_line 0,0 5,5
angle l0 l1 10 to 30
list
info l1
