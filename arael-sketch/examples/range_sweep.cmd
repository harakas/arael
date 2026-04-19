# Sweep range demo.
#
# Create an arc with a ~270 degree sweep, then constrain it to
# [30, 90]. The upper bound activates, pulling the sweep to 90.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_sweep.cmd

a0 = add_arc 0,0 3,3 -3,3
sweep a0 30 to 90
list
info a0
