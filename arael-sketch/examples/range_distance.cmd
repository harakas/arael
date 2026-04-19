# Range (inequality) dimension demo.
#
# Each bound is either a numeric literal or a live expression
# (anything non-numeric -- bare param name, entity property, compound
# arithmetic). Live bounds re-evaluate every solve so parameter
# changes propagate. Prefix with `=` to force a snapshot (evaluate
# once at command time, store the result as a literal).
#
# Run with:
#   cargo run -r -p arael-sketch -- --nogui --stdout --script \
#       arael-sketch/examples/range_distance.cmd
#
# Expected:
#  - First `distance`: literal lower bound, gap grows from 2 to 3.
#  - Second scenario: live range using params `low` and `high`. After
#    `param high 8`, the upper bound lifts and the gap can expand;
#    after `param low 9` (would violate low > high), the parameter
#    change is rejected by the solver-failure rollback.

add_line 0,0 4,0
add_line 0,2 4,3
distance L0 L1 >= 3.0
list
info L1

# Live range bounds tracked by user parameters.
param low 2
param high 5
add_line -5,-5 -1,-5
add_line -5,5 -1,5
distance L2 L3 low to high
list

param high 8
list
