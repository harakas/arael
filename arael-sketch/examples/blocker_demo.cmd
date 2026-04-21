# DOF-rejection blocker analysis demo.
# Adding coincident P0 P1 is already implied by hdistance=0 and
# vdistance=0 between the same two points; removal of either
# alone unblocks it.
add_point 2,2
add_point 2,2
hdistance P0 P1 0
vdistance P0 P1 0
coincident P0 P1
