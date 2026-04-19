# Horizontal-distance range demo.
#
# Two lines with p1 endpoints 10 units apart in x; constrain the
# |x|-distance between them to [2, 5]. The upper bound activates,
# pulling the endpoints together so the separation lands at 5.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_hdistance.cmd

l0 = add_line 0,0 3,0
l1 = add_line 10,5 15,5
hdistance l0.p1 l1.p1 2 to 5
list
