# Vertical-distance range demo.
#
# A nearly-horizontal line with a tiny initial y-slope; constrain
# |y|-distance between its endpoints to be at least 3. The bound
# activates, rotating the line so the endpoints separate vertically.
#
# Note: if the line started exactly horizontal (both y=0), the
# gradient of the |y|-distance residual is zero at that point and
# LM can't make progress. The tiny initial slope gives the solver
# a well-defined direction.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_vdistance.cmd

l0 = add_line 0,0 5,1
vdistance l0.p1 l0.p2 >= 3
list
info l0
