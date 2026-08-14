use super::*;

pub(crate) fn cmd_help(args: &str) -> CmdResult {
    if args.trim() == "full" {
        return Ok(CommandResult {
            output: include_str!("../../docs/COMMANDS.md").to_string(),
            is_error: false, no_echo: false, markdown: true,
        });
    }
    if args.is_empty() {
        Ok(ok("Commands: add_line add_point add_circle add_arc offset_line fillet chamfer split trim delete horizontal vertical \
            parallel perpendicular equal collinear tangent coincident concentric midpoint \
            symmetry point_on length radius sweep angle distance hdistance vdistance xangle freeze set_derived set_driven \
            lock unlock param del_param rename_param style select deselect print info list \
            find dof cost undo redo history goto center zoom cursor dim_pos clear let save load \
            exit help\n\
            Type 'help <command>' for details. 'help full' for complete reference."))
    } else {
        let msg = match args.trim() {
            "add_line" => "add_line x1,y1 x2,y2 [x3,y3 ...] [noconnect] [nocursor] [driven]",
            "add_rect" => "add_rect x1,y1 x2,y2 [hv] [noconnect] [noconstraint] [driven] [strict]",
            "add_rect3" => "add_rect3 p1 p2 p3 [noconnect] [noconstraint] [driven] [strict]",
            "add_rectcenter" => "add_rectcenter cx,cy px,py [hv] [noconnect] [noconstraint] [driven] [strict]",
            "add_point" => "add_point x,y [nocursor]",
            "add_circle" => "add_circle cx,cy radius [noconnect] [nocursor] [driven]",
            "add_circle2" => "add_circle2 p1 p2 [noconnect] [nocursor] [driven] — circle from 2 diametrically opposite points",
            "add_circle3" => "add_circle3 p1 p2 p3 [noconnect] [nocursor] [driven] — circle from 3 points on circumference",
            "add_circle2t" => "add_circle2t L0 L1 radius [noconnect] [noconstraint] [driven] [strict] — circle tangent to 2 lines",
            "add_circle3t" => "add_circle3t L0 L1 L2 [noconnect] [noconstraint] [driven] [strict] — circle tangent to 3 lines",
            "add_ellipse" => "add_ellipse cx,cy rx ry rotation_deg [noconnect] [nocursor] [driven]",
            "delete" => "delete <L0|P0|A0|EA0|C3|CL0H|d0> | delete L0 L1 parallel",
            "horizontal" => "horizontal L0 [L1 ...]",
            "vertical" => "vertical L0 [L1 ...]",
            "parallel" => "parallel L0 L1",
            "perpendicular" => "perpendicular L0 L1",
            "equal" => "equal L0 L1 (length) | equal A0 A1 (radius)",
            "collinear" => "collinear L0 L1",
            "tangent" => "tangent L0 A0 | tangent A0 A1",
            "coincident" => "coincident L0.p2 L1.p1 (any endpoint pair: P0, L0.p1/p2, A0.center/start/end)",
            "concentric" => "concentric A0 A1",
            "midpoint" => "midpoint P0 L0 | midpoint L0.p1 L1 | midpoint P0 A0 (arc angular midpoint)",
            "symmetry" => "symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry A0 L0 A1",
            "mirror" => "mirror L0 L1 ... about L_axis [noconstraint] [strict] | mirror selection about L_axis",
            "point_on" => "point_on P0 L0 | point_on L0.p1 A0",
            "length" => "length L0 5 | length L0 L0.length | length L0 =2*scale | length L0 {expr} [derived|driven]",
            "radius" => "radius A0 1.5 | radius A0 =5*scale | radius A0 {expr} [derived|driven]",
            "radius_b" => "radius_b A0 1.5 [derived|driven] -- ellipse semi-minor axis",
            "sweep" => "sweep A0 180 | sweep A0 =90*n | sweep A0 {expr} [derived|driven]",
            "angle" => "angle L0 L1 45 [supplement|closest|acute|obtuse] [derived|driven]",
            "distance" => "distance L0.p1 L1.p2 5 | distance P0 L0 3 | distance L0.p1 L1.p2 =expr [derived|driven]",
            "hdistance" => "hdistance L0.p1 L1.p2 5 [derived|driven] — horizontal (x-axis) distance",
            "vdistance" => "vdistance L0.p1 L1.p2 3 [derived|driven] — vertical (y-axis) distance",
            "xangle" => "xangle L0 45 [derived|driven] — line angle from x-axis in degrees",
            "freeze" => "freeze [L0 L1 A0 ...] — add numeric dimensions at current values (all if no args)",
            "set_derived" => "set_derived d0 (make dimension display-only)",
            "set_driven" => "set_driven d0 [value|\"expr\"] (make dimension constraining)",
            "lock" => "lock P0 | lock L0.p1 | lock L0.p1 x,y",
            "unlock" => "unlock P0 | unlock L0.p1",
            "param" => "param name value | param name \"expr\" (creates or updates)",
            "del_param" => "del_param name",
            "rename_param" => "rename_param old_name new_name",
            "style" => "style L0 [solid|dashed|dashdot]",
            "quiet" => "quiet L0 [on|off] — toggle/set quiet mode (hides dimensions and center)",
            "constr" => "constr L0 [on|off] — toggle/set construction line (dashdot, different color)",
            "drag" => "drag L0.p1 x,y | drag L0 @dx,dy — drag entity/endpoint to position",
            "select" => "select L0 [L1 ...] | select all | select L0 chain | select L0 linked",
            "deselect" => "deselect [L0 L1 ...] (clears all or specific)",
            "print" => "print <expression> (evaluate and display)",
            "info" => "info L0 | info P0 | info A0 | info d0 | info paramname",
            "measure" => "measure L0 | measure L0 L1 | measure P0 P1 | measure L0 A0",
            "list" => "list [all|lines|points|arcs|dims|params|constraints|constr|selection]",
            "find" => "find x,y [radius] (list nearby entities)",
            "undo" => "undo [n]",
            "redo" => "redo [n]",
            "history" => "history [n] (show last n entries)",
            "goto" => "goto <position> (jump to history position)",
            "center" => "center L0 | center x,y | center (fit all)",
            "zoom" => "zoom + | zoom - | zoom 2.0",
            "msg" => "msg text — print message to history (supports markdown, \\n for newlines)",
            "cursor" => "cursor [x,y | @dx,dy | on | off] — show/set/hide command cursor",
            "dim_pos" => "dim_pos d0 offset 1.5 | dim_pos d0 along 0.3 (@ for relative)",
            "clear" => "clear (new empty sketch)",
            "add_arc" => "add_arc x1,y1 x2,y2 xm,ym (start, end, midpoint)",
            "add_earc" => "add_earc p1 p2 rx ry rot_deg [large] [cw]",
            "add_earc3" => "add_earc3 p1 p2 pmid rx ry",
            "add_earc_center" => "add_earc_center cx,cy rx ry rot_deg start_deg end_deg [cw]",
            "add_earc_tangent" => "add_earc_tangent p1 t1 p2 t2 [bulge] (tangent-defined, bulge=perp_dist/half_chord)",
            "add_earc_rtangent" => "add_earc_rtangent p2 t2 [bulge] (chain from cursor+tangent)",
            "offset_line" | "offset" => "offset_line L0 distance (create parallel line offset by distance)",
            "fillet" => "fillet L1 L2 r [notangent] [noradius]  or  fillet L1.pN r [notangent] [noradius] (round a corner with a tangent arc of radius r; breaks the shared LL coincident, trims both lines, adds arc + tangent + radius dim)",
            "chamfer" => "chamfer L1 L2 d  or  chamfer L1.pN d (bevel a corner at distance d from the corner; breaks the shared LL coincident, trims both lines by d, adds a bevel line + corner anchor point + two equal distance dims)",
            "split" => "split L0 x,y [r]  or  split L0 by L1 L2... [nopin] (cut a line/arc at the intersections bracketing x,y, or at every crossing with the named cutters; pieces get new names, all constraints/dims/expressions transfer, cut endpoints are joined and pinned onto the cutter)",
            "trim" => "trim L0 x,y [r]  or  trim L0 by L1 L2  or  trim L0 by L1 forward|backward [nopin] (delete the span of a line/arc between crossings; same reference transfer as split; with no crossings the whole entity is deleted)",
            "let" => "let name = expression (session variable, scalar or coordinate)",
            "save" => "save path.json",
            "load" => "load path.json",
            "exit" | "quit" => "exit — close the application (blocked for MCP clients)",
            "dof" => "dof | dof analyze | dof eigenvalues [raw] | dof singular [raw] | dof jacobian",
            "cost" => "cost — print the current solver cost (sum of squared residuals)",
            "help" => "help | help <command> | help full",
            "explain" => "explain <constraint-command> [args] — dry-run: report accept/reject without changing the sketch",
            "perp" => "alias for perpendicular",
            other => return Err(format!("help: unknown command: {}. Usage: help | help <command> | help full", other).into()),
        };
        Ok(ok(msg.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Autocomplete
// ---------------------------------------------------------------------------

pub(crate) const COMMAND_NAMES: &[&str] = &[
    "add_line", "add_rect", "add_rect3", "add_rectcenter",
    "add_point", "add_circle", "add_circle2", "add_circle3", "add_circle2t", "add_circle3t", "add_ellipse",
    "add_arc", "add_earc", "add_earc3", "add_earc_center", "add_earc_tangent", "add_earc_rtangent", "offset_line", "offset", "fillet", "chamfer", "split", "trim",
    "delete", "horizontal", "vertical", "parallel", "perpendicular", "perp",
    "equal", "collinear", "tangent", "coincident", "concentric", "midpoint",
    "symmetry", "mirror", "point_on", "length", "radius", "radius_b", "sweep", "angle", "distance", "hdistance", "vdistance", "xangle",
    "set_derived", "set_driven",
    "lock", "unlock", "param", "del_param", "rename_param", "style", "quiet", "constr", "drag",
    "select", "deselect", "freeze", "print", "info", "measure", "list", "find", "let",
    "dof", "cost", "undo", "redo", "history", "goto", "center", "zoom",
    "cursor", "dim_pos", "clear", "save", "load", "help", "msg",
    "explain", "exit", "quit",
];

pub(crate) const GEO_FUNCTIONS: &[&str] = &[
    "intersect", "midpoint", "project", "along", "arc_point",
    "rotate", "mirror", "tangent", "normal", "dist", "angle",
];

// Named constants recognized by the expression parser (not functions).
pub(crate) const MATH_CONSTANTS: &[&str] = &["pi", "e"];

/// Generate autocomplete suggestions for the command input.
/// Returns completions for the word at `cursor_pos` in `input`.
pub fn complete(
    sketch: &Sketch,
    session_names: &HashMap<String, String>,
    input: &str,
    cursor_pos: usize,
) -> Vec<String> {
    // The caller's cursor is a byte position; floor it to a char
    // boundary so a cursor inside a multibyte char cannot panic.
    let mut end = cursor_pos.min(input.len());
    while end > 0 && !input.is_char_boundary(end) { end -= 1; }
    let input = &input[..end];
    let current_line = input.lines().last().unwrap_or("");
    let word_start = current_line.rfind(|c: char| c.is_whitespace()).map(|i| i + 1).unwrap_or(0);
    let current_word = &current_line[word_start..];
    let is_first_token = current_line[..word_start].trim().is_empty();

    // No completions when nothing typed on first token (would show all commands)
    if current_word.is_empty() && is_first_token { return Vec::new(); }

    // Dot completion (context-independent)
    if let Some(dot_pos) = current_word.rfind('.') {
        let before_dot = &current_word[..dot_pos];
        let after_dot = &current_word[dot_pos + 1..];
        let mut r = complete_after_dot(sketch, before_dot, after_dot);
        r.retain(|s| s != current_word);
        return r;
    }

    let mut results = Vec::new();

    // First token: command names only
    if is_first_token {
        add_matching(&mut results, current_word, COMMAND_NAMES);
        results.sort();
        results.dedup();
        results.truncate(20);
        return results;
    }

    // Non-first token: command-specific completions
    let first_cmd = current_line.split_whitespace().next().unwrap_or("");
    let token_index = current_line[..word_start].split_whitespace().count();
    // token_index: 1 = arg1, 2 = arg2, 3 = arg3

    // Collect already-completed args (excluding current word being typed)
    let typed_args: Vec<&str> = current_line[..word_start].split_whitespace().skip(1).collect();

    match first_cmd {
        // Variadic line commands: exclude already-typed lines
        "horizontal" | "vertical" => {
            add_lines_excluding(sketch, &mut results, current_word, &typed_args);
        }

        // Two-line commands: no suggestions after 2 args
        "parallel" | "perpendicular" | "perp" | "collinear" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Arc-only, exactly 2 args
        "concentric" => {
            if token_index <= 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Equal: match type of first arg, exactly 2 args
        "equal" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                let arg1 = current_line.split_whitespace().nth(1).unwrap_or("");
                if arg1.starts_with('L') {
                    add_lines(sketch, &mut results, current_word);
                } else if is_arc_name(arg1) {
                    add_arcs(sketch, &mut results, current_word);
                }
            }
        }

        // Tangent: line+arc or arc+arc, exactly 2 args
        "tangent" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Delete: entity / dimension / named constraint on first token,
        // plus the multi-token relational form (`delete L0 L1 parallel`)
        // offering entity completions and a constraint-type keyword
        // for the trailing slot.
        "delete" => {
            if token_index == 1 {
                add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
                add_dimensions(sketch, &mut results, current_word);
            } else if token_index <= 3 {
                add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
                add_matching(&mut results, current_word,
                    &["horizontal", "vertical", "parallel", "perpendicular",
                      "equal", "equal_length", "equal_radius", "collinear",
                      "tangent", "concentric", "coincident", "point_on",
                      "symmetry", "midpoint", "lock"]);
            }
        }
        "select" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["all"]);
            }
            add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
            if token_index == 2 {
                add_matching(&mut results, current_word, &["chain", "linked"]);
            }
        }

        // Single entity arg
        "info" | "center" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Endpoint commands: exactly 2 args
        "coincident" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }
        "lock" | "unlock" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Midpoint: arg1=point/endpoint, arg2=line
        "midpoint" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Point_on: arg1=point/endpoint, arg2=line or arc
        "point_on" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Symmetry: arg1=entity, arg2=line(mirror), arg3=entity
        "symmetry" => {
            if token_index <= 3 {
                if token_index == 2 {
                    add_lines(sketch, &mut results, current_word);
                } else {
                    add_all_entities(sketch, &mut results, current_word);
                }
            }
        }

        // Dimension: length (arg1=line, arg2=value/derived)
        "length" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: radius (arg1=arc, arg2=value/derived)
        "radius" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: radius_b (arg1=arc, arg2=value/derived) -- ellipse minor axis
        "radius_b" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: sweep (arg1=arc, arg2=value/derived)
        "sweep" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Freeze: lines and arcs
        "freeze" => {
            add_lines(sketch, &mut results, current_word);
            add_arcs(sketch, &mut results, current_word);
        }

        // Dimension: angle (arg1=line, arg2=line, arg3=value/derived)
        "angle" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: distance (arg1=endpoint, arg2=endpoint/line, arg3=value/derived)
        "distance" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // hdistance/vdistance: arg1=endpoint, arg2=endpoint, arg3=value/derived
        "hdistance" | "vdistance" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // xangle: arg1=line, arg2=value/derived
        "xangle" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension management: single dim arg
        "set_derived" | "set_driven" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            }
        }

        // dim_pos: arg1=dim, arg2=offset/along, arg3=value
        "dim_pos" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["offset", "along"]);
            }
        }

        // Style: arg1=entity, arg2=style value
        "style" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["solid", "dashed", "dashdot"]);
            }
        }

        // List: single filter arg
        "list" => {
            if token_index == 1 {
                add_matching(&mut results, current_word,
                    &["all", "lines", "points", "arcs", "dims", "params", "constraints", "constr", "selection",
                      "horizontal", "vertical", "parallel", "perpendicular", "equal", "collinear",
                      "tangent", "coincident", "concentric", "midpoint", "symmetry", "point_on", "lock",
                      "angle", "length", "radius", "sweep", "distance"]);
            }
        }

        // Help: single arg (full or command name)
        "help" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["full"]);
                add_matching(&mut results, current_word, COMMAND_NAMES);
            }
        }

        // Cursor: single keyword arg
        "cursor" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["on", "off", "show", "hide"]);
            }
        }


        // Param commands: single param name
        "param" | "del_param" | "rename_param" => {
            if token_index == 1 {
                add_params(sketch, &mut results, current_word);
            }
        }

        // Offset: arg1=line, arg2=expression
        "offset_line" | "offset" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else {
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Fillet: arg1=line-or-endpoint, arg2=line-or-radius, arg3=radius,
        // trailing=keywords. Line completions first, then expressions,
        // then the fixed keyword options.
        "fillet" => {
            match token_index {
                1 => add_lines(sketch, &mut results, current_word),
                2 => {
                    add_lines(sketch, &mut results, current_word);
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
                _ => {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                    for k in &["notangent", "noradius"] {
                        if k.starts_with(current_word) { results.push((*k).to_string()); }
                    }
                }
            }
        }

        // Chamfer: same arg shape as fillet, no keyword options yet.
        "chamfer" => {
            match token_index {
                1 => add_lines(sketch, &mut results, current_word),
                2 => {
                    add_lines(sketch, &mut results, current_word);
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
                _ => {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
        }

        // Split/Trim: arg1=target entity; then `by` + cutters, or a
        // coordinate; trailing keywords.
        "split" | "trim" => {
            match token_index {
                1 => {
                    add_lines(sketch, &mut results, current_word);
                    add_arcs(sketch, &mut results, current_word);
                }
                2 => {
                    add_matching(&mut results, current_word, &["by"]);
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
                _ => {
                    add_lines(sketch, &mut results, current_word);
                    add_arcs(sketch, &mut results, current_word);
                    let kws: &[&str] = if first_cmd == "trim" {
                        &["forward", "backward", "nopin"]
                    } else {
                        &["nopin"]
                    };
                    for k in kws {
                        if k.starts_with(current_word) { results.push((*k).to_string()); }
                    }
                }
            }
        }

        // Geometry creation: position-aware completions
        // add_line: [coord1] [coord2] [noconnect] [nocursor]
        // add_point: [coord] (no flags)
        // add_circle: [center] [radius] [noconnect] [nocursor]
        // add_arc: [start] [end] [mid] [noconnect] [nocursor]
        "add_line" => {
            let line_kws = ["noconnect", "nocursor", "notangent", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !line_kws.contains(a)).count();
            if coord_args < 2 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 1 {
                for kw in &line_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_rect" | "add_rectcenter" => {
            let rect_kws = ["hv", "noconnect", "noconstraint", "driven", "strict"];
            let coord_args = typed_args.iter().filter(|a| !rect_kws.contains(a)).count();
            if coord_args < 2 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 2 {
                for kw in &rect_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_rect3" => {
            let rect_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let coord_args = typed_args.iter().filter(|a| !rect_kws.contains(a)).count();
            if coord_args < 3 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 3 {
                for kw in &rect_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_point" => {
            let coord_args = typed_args.iter().filter(|a| **a != "nocursor").count();
            if coord_args < 1 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 1 && !typed_args.contains(&"nocursor") {
                add_matching(&mut results, current_word, &["nocursor"]);
            }
        }
        "add_circle" => {
            let circle_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !circle_kws.contains(a)).count();
            if coord_args < 2 {
                if coord_args == 0 {
                    add_matching(&mut results, current_word, &["cursor"]);
                    add_all_entities(sketch, &mut results, current_word);
                    add_session_names(session_names, &mut results, current_word);
                } else {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
            if coord_args >= 2 {
                for kw in &circle_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle2" | "add_circle3" => {
            let circle_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let max_coords = if first_cmd == "add_circle2" { 2 } else { 3 };
            let coord_args = typed_args.iter().filter(|a| !circle_kws.contains(a)).count();
            if coord_args < max_coords {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= max_coords {
                for kw in &circle_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle2t" => {
            let ct_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let non_kw_args = typed_args.iter().filter(|a| !ct_kws.contains(a)).count();
            if non_kw_args < 2 {
                add_lines(sketch, &mut results, current_word);
            } else if non_kw_args == 2 {
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
            if non_kw_args >= 3 {
                for kw in &ct_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle3t" => {
            let ct_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let non_kw_args = typed_args.iter().filter(|a| !ct_kws.contains(a)).count();
            if non_kw_args < 3 {
                add_lines(sketch, &mut results, current_word);
            }
            if non_kw_args >= 3 {
                for kw in &ct_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_ellipse" => {
            let ellipse_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !ellipse_kws.contains(a)).count();
            if coord_args < 4 {
                if coord_args == 0 {
                    add_matching(&mut results, current_word, &["cursor"]);
                    add_all_entities(sketch, &mut results, current_word);
                    add_session_names(session_names, &mut results, current_word);
                } else {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
            if coord_args >= 4 {
                for kw in &ellipse_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "mirror" => {
            let has_about = typed_args.contains(&"about");
            if !has_about {
                // Before "about": offer entities and "selection"/"about"
                add_matching(&mut results, current_word, &["selection", "about"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            } else {
                // After "about": first the mirror line, then keywords
                let after_about_count = typed_args.iter().skip_while(|&&a| a != "about").skip(1)
                    .filter(|&&a| a != "noconstraint" && a != "strict").count();
                if after_about_count == 0 {
                    add_lines(sketch, &mut results, current_word);
                } else {
                    let mirror_kws = ["noconstraint", "strict"];
                    for kw in &mirror_kws {
                        if !typed_args.contains(kw) {
                            add_matching(&mut results, current_word, &[kw]);
                        }
                    }
                }
            }
        }
        "add_arc" => {
            let arc_kws = ["noconnect", "nocursor", "notangent", "quiet", "driven"];
            let coord_args = typed_args.iter().filter(|a| !arc_kws.contains(a)).count();
            if coord_args < 3 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 3 {
                for kw in &arc_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }

        // Expression-only: print, let
        "print" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }
        "let" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }

        // No completions for these
        // dof: single arg
        "dof" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["analyze"]);
            }
        }

        "undo" | "redo" | "history" | "goto" | "cost" | "clear"
        | "deselect" | "save" | "load" | "msg" | "find" | "zoom" => {}

        _ => {}
    }

    results.sort();
    results.dedup();
    results.truncate(20);
    results
}

// --- Completion helpers ---

pub(crate) fn add_matching(results: &mut Vec<String>, prefix: &str, candidates: &[&str]) {
    for &c in candidates {
        if c.starts_with(prefix) && c != prefix {
            results.push(c.to_string());
        }
    }
}

pub(crate) fn add_lines(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

pub(crate) fn add_arcs(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

pub(crate) fn add_points(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

pub(crate) fn add_all_entities(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    add_lines(sketch, results, prefix);
    add_points(sketch, results, prefix);
    add_arcs(sketch, results, prefix);
}

pub(crate) fn add_lines_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

pub(crate) fn add_all_entities_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix && !exclude.contains(&p.name.as_str()) {
            results.push(p.name.clone());
        }
    }
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

pub(crate) fn add_dimensions(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for d in &sketch.dimensions {
        if d.name.starts_with(prefix) && d.name != prefix {
            results.push(d.name.clone());
        }
    }
}

pub(crate) fn add_params(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for p in &sketch.user_params {
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

pub(crate) fn add_session_names(session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    for name in session_names.keys() {
        if name == "_" { continue; }
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

pub(crate) fn add_expression_completions(sketch: &Sketch, session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    add_dimensions(sketch, results, prefix);
    add_params(sketch, results, prefix);
    add_session_names(session_names, results, prefix);
    add_matching(results, prefix, GEO_FUNCTIONS);
    // Math functions come from arael-sym's authoritative table.
    for name in arael_sym::function_names() {
        if name.starts_with(prefix) && name != prefix {
            results.push(name.to_string());
        }
    }
    add_matching(results, prefix, MATH_CONSTANTS);
}

/// Complete after a dot: "L0." → ["p1", "p2"], "A0." → ["center", "start", "end"], etc.
pub(crate) fn complete_after_dot(sketch: &Sketch, before_dot: &str, after_dot: &str) -> Vec<String> {
    let mut results = Vec::new();

    // Check for double-dot: "L0.p1." → x, y
    if let Some(first_dot) = before_dot.rfind('.') {
        let entity = &before_dot[..first_dot];
        let prop = &before_dot[first_dot + 1..];
        // L<N>.p1. or L<N>.p2. → x, y
        if entity.starts_with('L') && (prop == "p1" || prop == "p2") {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // A<N>.center. → x, y
        if is_arc_name(entity) && prop == "center" {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // P<N>. after P<N>.pos or similar
        return results;
    }

    // Single dot: entity.suffix
    if before_dot.starts_with('L') && sketch.lines.refs().any(|r| sketch.lines[r].name == before_dot) {
        for &s in &["p1", "p2", "length", "angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if is_arc_name(before_dot) && sketch.arcs.refs().any(|r| sketch.arcs[r].name == before_dot) {
        for &s in &["center", "start", "end", "radius", "start_angle", "end_angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if before_dot.starts_with('P') && sketch.points.refs().any(|r| sketch.points[r].name == before_dot) {
        for &s in &["x", "y"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

