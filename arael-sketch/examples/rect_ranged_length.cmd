# Rectangle with ranged width and height via `length`.
#
# Same as rect_ranged.cmd but uses the `length` command directly on
# the edges rather than point-point distance. Reads more naturally
# when you care about the length of a specific edge.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/rect_ranged_length.cmd

l0,l1,l2,l3 = add_rect 0,0 5,3
length l0 4 to 6
length l1 1 to 4
list
info l0
info l1
