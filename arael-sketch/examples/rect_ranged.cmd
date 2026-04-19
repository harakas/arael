# 5x3 rectangle with ranged width and height.
#
# Lines created by add_rect form a cycle: bottom, right, top, left.
# The multi-assignment captures the four freshly-made line names
# into session variables so the script doesn't depend on which IDs
# the rectangle ends up with (if prior geometry exists the names
# would shift).
#
# Without `hv`, add_rect adds parallel + perpendicular constraints
# but doesn't pin orientation, so the rectangle can rotate to
# satisfy the ranges if they become active.
#
# - Width range 4..6: measured via endpoint-to-endpoint distance on
#   the bottom edge. Initial 5 is inside [4, 6], bound inactive.
# - Height range 1..4: endpoint-to-endpoint on the right edge.
#   Initial 3 is inside [1, 4], bound inactive.
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/rect_ranged.cmd

l0,l1,l2,l3 = add_rect 0,0 5,3
distance l0.p1 l0.p2 4 to 6
distance l1.p1 l1.p2 1 to 4
list
info l0
info l1
