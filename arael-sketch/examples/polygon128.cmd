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
equal L0 L1
equal L1 L2
equal L2 L3
equal L3 L4
equal L4 L5
equal L5 L6
equal L6 L7
equal L7 L8
equal L8 L9
equal L9 L10
equal L10 L11
equal L11 L12
equal L12 L13
equal L13 L14
equal L14 L15
equal L15 L16
equal L16 L17
equal L17 L18
equal L18 L19
equal L19 L20
equal L20 L21
equal L21 L22
equal L22 L23
equal L23 L24
equal L24 L25
equal L25 L26
equal L26 L27
equal L27 L28
equal L28 L29
equal L29 L30
equal L30 L31
equal L31 L32
equal L32 L33
equal L33 L34
equal L34 L35
equal L35 L36
equal L36 L37
equal L37 L38
equal L38 L39
equal L39 L40
equal L40 L41
equal L41 L42
equal L42 L43
equal L43 L44
equal L44 L45
equal L45 L46
equal L46 L47
equal L47 L48
equal L48 L49
equal L49 L50
equal L50 L51
equal L51 L52
equal L52 L53
equal L53 L54
equal L54 L55
equal L55 L56
equal L56 L57
equal L57 L58
equal L58 L59
equal L59 L60
equal L60 L61
equal L61 L62
equal L62 L63
equal L63 L64
equal L64 L65
equal L65 L66
equal L66 L67
equal L67 L68
equal L68 L69
equal L69 L70
equal L70 L71
equal L71 L72
equal L72 L73
equal L73 L74
equal L74 L75
equal L75 L76
equal L76 L77
equal L77 L78
equal L78 L79
equal L79 L80
equal L80 L81
equal L81 L82
equal L82 L83
equal L83 L84
equal L84 L85
equal L85 L86
equal L86 L87
equal L87 L88
equal L88 L89
equal L89 L90
equal L90 L91
equal L91 L92
equal L92 L93
equal L93 L94
equal L94 L95
equal L95 L96
equal L96 L97
equal L97 L98
equal L98 L99
equal L99 L100
equal L100 L101
equal L101 L102
equal L102 L103
equal L103 L104
equal L104 L105
equal L105 L106
equal L106 L107
equal L107 L108
equal L108 L109
equal L109 L110
equal L110 L111
equal L111 L112
equal L112 L113
equal L113 L114
equal L114 L115
equal L115 L116
equal L116 L117
equal L117 L118
equal L118 L119
equal L119 L120
equal L120 L121
equal L121 L122
equal L122 L123
equal L123 L124
equal L124 L125
equal L125 L126
equal L126 L127

# Phase 4: Equal-radius constraints (chain)
equal A0 A1
equal A1 A2
equal A2 A3
equal A3 A4
equal A4 A5
equal A5 A6
equal A6 A7
equal A7 A8
equal A8 A9
equal A9 A10
equal A10 A11
equal A11 A12
equal A12 A13
equal A13 A14
equal A14 A15
equal A15 A16
equal A16 A17
equal A17 A18
equal A18 A19
equal A19 A20
equal A20 A21
equal A21 A22
equal A22 A23
equal A23 A24
equal A24 A25
equal A25 A26
equal A26 A27
equal A27 A28
equal A28 A29
equal A29 A30
equal A30 A31
equal A31 A32
equal A32 A33
equal A33 A34
equal A34 A35
equal A35 A36
equal A36 A37
equal A37 A38
equal A38 A39
equal A39 A40
equal A40 A41
equal A41 A42
equal A42 A43
equal A43 A44
equal A44 A45
equal A45 A46
equal A46 A47
equal A47 A48
equal A48 A49
equal A49 A50
equal A50 A51
equal A51 A52
equal A52 A53
equal A53 A54
equal A54 A55
equal A55 A56
equal A56 A57
equal A57 A58
equal A58 A59
equal A59 A60
equal A60 A61
equal A61 A62
equal A62 A63
equal A63 A64
equal A64 A65
equal A65 A66
equal A66 A67
equal A67 A68
equal A68 A69
equal A69 A70
equal A70 A71
equal A71 A72
equal A72 A73
equal A73 A74
equal A74 A75
equal A75 A76
equal A76 A77
equal A77 A78
equal A78 A79
equal A79 A80
equal A80 A81
equal A81 A82
equal A82 A83
equal A83 A84
equal A84 A85
equal A85 A86
equal A86 A87
equal A87 A88
equal A88 A89
equal A89 A90
equal A90 A91
equal A91 A92
equal A92 A93
equal A93 A94
equal A94 A95
equal A95 A96
equal A96 A97
equal A97 A98
equal A98 A99
equal A99 A100
equal A100 A101
equal A101 A102
equal A102 A103
equal A103 A104
equal A104 A105
equal A105 A106
equal A106 A107
equal A107 A108
equal A108 A109
equal A109 A110
equal A110 A111
equal A111 A112
equal A112 A113
equal A113 A114
equal A114 A115
equal A115 A116
equal A116 A117
equal A117 A118
equal A118 A119
equal A119 A120
equal A120 A121
equal A121 A122
equal A122 A123
equal A123 A124
equal A124 A125
equal A125 A126
equal A126 A127

# Phase 5: Dimensions
length L0 2.5
radius A0 0.5
