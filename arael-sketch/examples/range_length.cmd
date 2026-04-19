# Length range demo.
#
# Create a line of length 10, then constrain it to [4, 6]. The upper
# bound activates, pulling the endpoints in so the length lands at 6.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_length.cmd

l0 = add_line 0,0 10,0
length l0 4 to 6
list
info l0
