# polygon128.cmd -- Benchmark script: 128-segment polygon with circles
#
# Creates a 128-segment closed polygon inscribed in a circle of radius 50,
# with a circle at every vertex. All segments are constrained to equal length,
# all circles to equal radius, then dimensions set the segment length and
# circle radius. Total: 512 commands (128 lines + 128 circles + 127 equal
# length + 127 equal radius + 2 dimensions).
#
# Run:
#   cargo run -p arael-sketch --release -- --empty --script arael-sketch/examples/polygon128.cmd
#
# Or from a release binary:
#   arael-sketch --empty --script examples/polygon128.cmd

# Phase 1: Create 128 lines forming a polygon inscribed in r=50 circle
add_line 50.00000,0.00000 49.93977,2.45338
add_line 49.75924,4.90086
add_line 49.45883,7.33652
add_line 49.03926,9.75452
add_line 48.50156,12.14901
add_line 47.84702,14.51423
add_line 47.07720,16.84449
add_line 46.19398,19.13417
add_line 45.19946,21.37775
add_line 44.09606,23.56984
add_line 42.88643,25.70514
add_line 41.57348,27.77851
add_line 40.16038,29.78497
add_line 38.65052,31.71966
add_line 37.04756,33.57795
add_line 35.35534,35.35534
add_line 33.57795,37.04756
add_line 31.71966,38.65052
add_line 29.78497,40.16038
add_line 27.77851,41.57348
add_line 25.70514,42.88643
add_line 23.56984,44.09606
add_line 21.37775,45.19946
add_line 19.13417,46.19398
add_line 16.84449,47.07720
add_line 14.51423,47.84702
add_line 12.14901,48.50156
add_line 9.75452,49.03926
add_line 7.33652,49.45883
add_line 4.90086,49.75924
add_line 2.45338,49.93977
add_line 0.00000,50.00000
add_line -2.45338,49.93977
add_line -4.90086,49.75924
add_line -7.33652,49.45883
add_line -9.75452,49.03926
add_line -12.14901,48.50156
add_line -14.51423,47.84702
add_line -16.84449,47.07720
add_line -19.13417,46.19398
add_line -21.37775,45.19946
add_line -23.56984,44.09606
add_line -25.70514,42.88643
add_line -27.77851,41.57348
add_line -29.78497,40.16038
add_line -31.71966,38.65052
add_line -33.57795,37.04756
add_line -35.35534,35.35534
add_line -37.04756,33.57795
add_line -38.65052,31.71966
add_line -40.16038,29.78497
add_line -41.57348,27.77851
add_line -42.88643,25.70514
add_line -44.09606,23.56984
add_line -45.19946,21.37775
add_line -46.19398,19.13417
add_line -47.07720,16.84449
add_line -47.84702,14.51423
add_line -48.50156,12.14901
add_line -49.03926,9.75452
add_line -49.45883,7.33652
add_line -49.75924,4.90086
add_line -49.93977,2.45338
add_line -50.00000,0.00000
add_line -49.93977,-2.45338
add_line -49.75924,-4.90086
add_line -49.45883,-7.33652
add_line -49.03926,-9.75452
add_line -48.50156,-12.14901
add_line -47.84702,-14.51423
add_line -47.07720,-16.84449
add_line -46.19398,-19.13417
add_line -45.19946,-21.37775
add_line -44.09606,-23.56984
add_line -42.88643,-25.70514
add_line -41.57348,-27.77851
add_line -40.16038,-29.78497
add_line -38.65052,-31.71966
add_line -37.04756,-33.57795
add_line -35.35534,-35.35534
add_line -33.57795,-37.04756
add_line -31.71966,-38.65052
add_line -29.78497,-40.16038
add_line -27.77851,-41.57348
add_line -25.70514,-42.88643
add_line -23.56984,-44.09606
add_line -21.37775,-45.19946
add_line -19.13417,-46.19398
add_line -16.84449,-47.07720
add_line -14.51423,-47.84702
add_line -12.14901,-48.50156
add_line -9.75452,-49.03926
add_line -7.33652,-49.45883
add_line -4.90086,-49.75924
add_line -2.45338,-49.93977
add_line -0.00000,-50.00000
add_line 2.45338,-49.93977
add_line 4.90086,-49.75924
add_line 7.33652,-49.45883
add_line 9.75452,-49.03926
add_line 12.14901,-48.50156
add_line 14.51423,-47.84702
add_line 16.84449,-47.07720
add_line 19.13417,-46.19398
add_line 21.37775,-45.19946
add_line 23.56984,-44.09606
add_line 25.70514,-42.88643
add_line 27.77851,-41.57348
add_line 29.78497,-40.16038
add_line 31.71966,-38.65052
add_line 33.57795,-37.04756
add_line 35.35534,-35.35534
add_line 37.04756,-33.57795
add_line 38.65052,-31.71966
add_line 40.16038,-29.78497
add_line 41.57348,-27.77851
add_line 42.88643,-25.70514
add_line 44.09606,-23.56984
add_line 45.19946,-21.37775
add_line 46.19398,-19.13417
add_line 47.07720,-16.84449
add_line 47.84702,-14.51423
add_line 48.50156,-12.14901
add_line 49.03926,-9.75452
add_line 49.45883,-7.33652
add_line 49.75924,-4.90086
add_line 49.93977,-2.45338
add_line L0.p1

# Phase 2: Create 128 circles at each vertex (auto-connects to line endpoints)
add_circle 50.00000,0.00000 1
add_circle 49.93977,2.45338 1
add_circle 49.75924,4.90086 1
add_circle 49.45883,7.33652 1
add_circle 49.03926,9.75452 1
add_circle 48.50156,12.14901 1
add_circle 47.84702,14.51423 1
add_circle 47.07720,16.84449 1
add_circle 46.19398,19.13417 1
add_circle 45.19946,21.37775 1
add_circle 44.09606,23.56984 1
add_circle 42.88643,25.70514 1
add_circle 41.57348,27.77851 1
add_circle 40.16038,29.78497 1
add_circle 38.65052,31.71966 1
add_circle 37.04756,33.57795 1
add_circle 35.35534,35.35534 1
add_circle 33.57795,37.04756 1
add_circle 31.71966,38.65052 1
add_circle 29.78497,40.16038 1
add_circle 27.77851,41.57348 1
add_circle 25.70514,42.88643 1
add_circle 23.56984,44.09606 1
add_circle 21.37775,45.19946 1
add_circle 19.13417,46.19398 1
add_circle 16.84449,47.07720 1
add_circle 14.51423,47.84702 1
add_circle 12.14901,48.50156 1
add_circle 9.75452,49.03926 1
add_circle 7.33652,49.45883 1
add_circle 4.90086,49.75924 1
add_circle 2.45338,49.93977 1
add_circle 0.00000,50.00000 1
add_circle -2.45338,49.93977 1
add_circle -4.90086,49.75924 1
add_circle -7.33652,49.45883 1
add_circle -9.75452,49.03926 1
add_circle -12.14901,48.50156 1
add_circle -14.51423,47.84702 1
add_circle -16.84449,47.07720 1
add_circle -19.13417,46.19398 1
add_circle -21.37775,45.19946 1
add_circle -23.56984,44.09606 1
add_circle -25.70514,42.88643 1
add_circle -27.77851,41.57348 1
add_circle -29.78497,40.16038 1
add_circle -31.71966,38.65052 1
add_circle -33.57795,37.04756 1
add_circle -35.35534,35.35534 1
add_circle -37.04756,33.57795 1
add_circle -38.65052,31.71966 1
add_circle -40.16038,29.78497 1
add_circle -41.57348,27.77851 1
add_circle -42.88643,25.70514 1
add_circle -44.09606,23.56984 1
add_circle -45.19946,21.37775 1
add_circle -46.19398,19.13417 1
add_circle -47.07720,16.84449 1
add_circle -47.84702,14.51423 1
add_circle -48.50156,12.14901 1
add_circle -49.03926,9.75452 1
add_circle -49.45883,7.33652 1
add_circle -49.75924,4.90086 1
add_circle -49.93977,2.45338 1
add_circle -50.00000,0.00000 1
add_circle -49.93977,-2.45338 1
add_circle -49.75924,-4.90086 1
add_circle -49.45883,-7.33652 1
add_circle -49.03926,-9.75452 1
add_circle -48.50156,-12.14901 1
add_circle -47.84702,-14.51423 1
add_circle -47.07720,-16.84449 1
add_circle -46.19398,-19.13417 1
add_circle -45.19946,-21.37775 1
add_circle -44.09606,-23.56984 1
add_circle -42.88643,-25.70514 1
add_circle -41.57348,-27.77851 1
add_circle -40.16038,-29.78497 1
add_circle -38.65052,-31.71966 1
add_circle -37.04756,-33.57795 1
add_circle -35.35534,-35.35534 1
add_circle -33.57795,-37.04756 1
add_circle -31.71966,-38.65052 1
add_circle -29.78497,-40.16038 1
add_circle -27.77851,-41.57348 1
add_circle -25.70514,-42.88643 1
add_circle -23.56984,-44.09606 1
add_circle -21.37775,-45.19946 1
add_circle -19.13417,-46.19398 1
add_circle -16.84449,-47.07720 1
add_circle -14.51423,-47.84702 1
add_circle -12.14901,-48.50156 1
add_circle -9.75452,-49.03926 1
add_circle -7.33652,-49.45883 1
add_circle -4.90086,-49.75924 1
add_circle -2.45338,-49.93977 1
add_circle -0.00000,-50.00000 1
add_circle 2.45338,-49.93977 1
add_circle 4.90086,-49.75924 1
add_circle 7.33652,-49.45883 1
add_circle 9.75452,-49.03926 1
add_circle 12.14901,-48.50156 1
add_circle 14.51423,-47.84702 1
add_circle 16.84449,-47.07720 1
add_circle 19.13417,-46.19398 1
add_circle 21.37775,-45.19946 1
add_circle 23.56984,-44.09606 1
add_circle 25.70514,-42.88643 1
add_circle 27.77851,-41.57348 1
add_circle 29.78497,-40.16038 1
add_circle 31.71966,-38.65052 1
add_circle 33.57795,-37.04756 1
add_circle 35.35534,-35.35534 1
add_circle 37.04756,-33.57795 1
add_circle 38.65052,-31.71966 1
add_circle 40.16038,-29.78497 1
add_circle 41.57348,-27.77851 1
add_circle 42.88643,-25.70514 1
add_circle 44.09606,-23.56984 1
add_circle 45.19946,-21.37775 1
add_circle 46.19398,-19.13417 1
add_circle 47.07720,-16.84449 1
add_circle 47.84702,-14.51423 1
add_circle 48.50156,-12.14901 1
add_circle 49.03926,-9.75452 1
add_circle 49.45883,-7.33652 1
add_circle 49.75924,-4.90086 1
add_circle 49.93977,-2.45338 1

# Phase 3: Equal-length constraints (chain)
equal L0 L1 force
equal L1 L2 force
equal L2 L3 force
equal L3 L4 force
equal L4 L5 force
equal L5 L6 force
equal L6 L7 force
equal L7 L8 force
equal L8 L9 force
equal L9 L10 force
equal L10 L11 force
equal L11 L12 force
equal L12 L13 force
equal L13 L14 force
equal L14 L15 force
equal L15 L16 force
equal L16 L17 force
equal L17 L18 force
equal L18 L19 force
equal L19 L20 force
equal L20 L21 force
equal L21 L22 force
equal L22 L23 force
equal L23 L24 force
equal L24 L25 force
equal L25 L26 force
equal L26 L27 force
equal L27 L28 force
equal L28 L29 force
equal L29 L30 force
equal L30 L31 force
equal L31 L32 force
equal L32 L33 force
equal L33 L34 force
equal L34 L35 force
equal L35 L36 force
equal L36 L37 force
equal L37 L38 force
equal L38 L39 force
equal L39 L40 force
equal L40 L41 force
equal L41 L42 force
equal L42 L43 force
equal L43 L44 force
equal L44 L45 force
equal L45 L46 force
equal L46 L47 force
equal L47 L48 force
equal L48 L49 force
equal L49 L50 force
equal L50 L51 force
equal L51 L52 force
equal L52 L53 force
equal L53 L54 force
equal L54 L55 force
equal L55 L56 force
equal L56 L57 force
equal L57 L58 force
equal L58 L59 force
equal L59 L60 force
equal L60 L61 force
equal L61 L62 force
equal L62 L63 force
equal L63 L64 force
equal L64 L65 force
equal L65 L66 force
equal L66 L67 force
equal L67 L68 force
equal L68 L69 force
equal L69 L70 force
equal L70 L71 force
equal L71 L72 force
equal L72 L73 force
equal L73 L74 force
equal L74 L75 force
equal L75 L76 force
equal L76 L77 force
equal L77 L78 force
equal L78 L79 force
equal L79 L80 force
equal L80 L81 force
equal L81 L82 force
equal L82 L83 force
equal L83 L84 force
equal L84 L85 force
equal L85 L86 force
equal L86 L87 force
equal L87 L88 force
equal L88 L89 force
equal L89 L90 force
equal L90 L91 force
equal L91 L92 force
equal L92 L93 force
equal L93 L94 force
equal L94 L95 force
equal L95 L96 force
equal L96 L97 force
equal L97 L98 force
equal L98 L99 force
equal L99 L100 force
equal L100 L101 force
equal L101 L102 force
equal L102 L103 force
equal L103 L104 force
equal L104 L105 force
equal L105 L106 force
equal L106 L107 force
equal L107 L108 force
equal L108 L109 force
equal L109 L110 force
equal L110 L111 force
equal L111 L112 force
equal L112 L113 force
equal L113 L114 force
equal L114 L115 force
equal L115 L116 force
equal L116 L117 force
equal L117 L118 force
equal L118 L119 force
equal L119 L120 force
equal L120 L121 force
equal L121 L122 force
equal L122 L123 force
equal L123 L124 force
equal L124 L125 force
equal L125 L126 force
equal L126 L127 force

# Phase 4: Equal-radius constraints (chain)
equal A0 A1 force
equal A1 A2 force
equal A2 A3 force
equal A3 A4 force
equal A4 A5 force
equal A5 A6 force
equal A6 A7 force
equal A7 A8 force
equal A8 A9 force
equal A9 A10 force
equal A10 A11 force
equal A11 A12 force
equal A12 A13 force
equal A13 A14 force
equal A14 A15 force
equal A15 A16 force
equal A16 A17 force
equal A17 A18 force
equal A18 A19 force
equal A19 A20 force
equal A20 A21 force
equal A21 A22 force
equal A22 A23 force
equal A23 A24 force
equal A24 A25 force
equal A25 A26 force
equal A26 A27 force
equal A27 A28 force
equal A28 A29 force
equal A29 A30 force
equal A30 A31 force
equal A31 A32 force
equal A32 A33 force
equal A33 A34 force
equal A34 A35 force
equal A35 A36 force
equal A36 A37 force
equal A37 A38 force
equal A38 A39 force
equal A39 A40 force
equal A40 A41 force
equal A41 A42 force
equal A42 A43 force
equal A43 A44 force
equal A44 A45 force
equal A45 A46 force
equal A46 A47 force
equal A47 A48 force
equal A48 A49 force
equal A49 A50 force
equal A50 A51 force
equal A51 A52 force
equal A52 A53 force
equal A53 A54 force
equal A54 A55 force
equal A55 A56 force
equal A56 A57 force
equal A57 A58 force
equal A58 A59 force
equal A59 A60 force
equal A60 A61 force
equal A61 A62 force
equal A62 A63 force
equal A63 A64 force
equal A64 A65 force
equal A65 A66 force
equal A66 A67 force
equal A67 A68 force
equal A68 A69 force
equal A69 A70 force
equal A70 A71 force
equal A71 A72 force
equal A72 A73 force
equal A73 A74 force
equal A74 A75 force
equal A75 A76 force
equal A76 A77 force
equal A77 A78 force
equal A78 A79 force
equal A79 A80 force
equal A80 A81 force
equal A81 A82 force
equal A82 A83 force
equal A83 A84 force
equal A84 A85 force
equal A85 A86 force
equal A86 A87 force
equal A87 A88 force
equal A88 A89 force
equal A89 A90 force
equal A90 A91 force
equal A91 A92 force
equal A92 A93 force
equal A93 A94 force
equal A94 A95 force
equal A95 A96 force
equal A96 A97 force
equal A97 A98 force
equal A98 A99 force
equal A99 A100 force
equal A100 A101 force
equal A101 A102 force
equal A102 A103 force
equal A103 A104 force
equal A104 A105 force
equal A105 A106 force
equal A106 A107 force
equal A107 A108 force
equal A108 A109 force
equal A109 A110 force
equal A110 A111 force
equal A111 A112 force
equal A112 A113 force
equal A113 A114 force
equal A114 A115 force
equal A115 A116 force
equal A116 A117 force
equal A117 A118 force
equal A118 A119 force
equal A119 A120 force
equal A120 A121 force
equal A121 A122 force
equal A122 A123 force
equal A123 A124 force
equal A124 A125 force
equal A125 A126 force
equal A126 A127 force

# Phase 5: Dimensions
length L0 2.5 force
radius A0 0.5 force
