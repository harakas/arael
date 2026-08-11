// The history invariant behind every editing command: undo returns
// the sketch to the state before the command, redo returns it to the
// state after, and a rejected or read-only command changes nothing.
// The missing-history bugs (drag, quiet, constr, delete-relational,
// auto-tangent -- W4) were all violations of exactly this property.
//
// State is compared as a structural fingerprint (constraints,
// dimensions, entities, flags) plus the parameter vector within a
// tolerance: History::undo re-solves the restored snapshot, and a
// solve from an already-converged state may move parameters by a
// convergence-tolerance step, not more.

use arael::simple_lm::RootProblem;
use arael_sketch_backend::commands::{execute, CommandContext};

#[derive(PartialEq, Debug, Clone)]
struct Fingerprint {
    constraints: Vec<String>,
    dimensions: Vec<String>,
    points: Vec<String>,
    lines: Vec<(String, bool, bool)>,
    arcs: Vec<(String, bool, bool)>,
    user_params: Vec<String>,
}

fn fingerprint(ctx: &CommandContext) -> Fingerprint {
    let s = &ctx.sketch;
    let mut constraints = s.list_constraints();
    constraints.sort();
    Fingerprint {
        constraints,
        dimensions: s.dimensions.iter()
            .map(|d| format!("{} did={} derived={} expr={:?} value={:.4}",
                d.name, d.did, d.derived, d.expr_str, d.value))
            .collect(),
        points: s.points.refs().map(|r| s.points[r].name.clone()).collect(),
        lines: s.lines.refs()
            .map(|r| { let l = &s.lines[r]; (l.name.clone(), l.construction, l.quiet) })
            .collect(),
        arcs: s.arcs.refs()
            .map(|r| { let a = &s.arcs[r]; (a.name.clone(), a.construction, a.quiet) })
            .collect(),
        user_params: s.user_params.iter()
            .map(|p| format!("{}={}", p.name, p.expr_str)).collect(),
    }
}

fn params_of(ctx: &mut CommandContext) -> Vec<f64> {
    let mut p = Vec::new();
    ctx.sketch.mutate_values(|s| s.serialize(&mut p));
    p
}

fn params_close(a: &[f64], b: &[f64], what: &str, cmd: &str) {
    assert_eq!(a.len(), b.len(), "{}: param count changed after {}", what, cmd);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-4,
            "{}: param {} differs after {}: {} vs {}", what, i, cmd, x, y);
    }
}

fn run_one(ctx: &mut CommandContext, cmd: &str) -> bool {
    let results = execute(ctx, cmd);
    !results.iter().any(|r| r.is_error)
}

#[test]
fn undo_redo_is_identity_for_every_command() {
    let mut ctx = CommandContext::new();
    assert!(run_one(&mut ctx, "add_rect 0,0 10,6"));
    assert!(run_one(&mut ctx, "add_circle 20,3 2 noconnect"));
    assert!(run_one(&mut ctx, "add_line 12,0 18,0 noconnect"));
    assert!(run_one(&mut ctx, "add_point 15,8"));

    // Editing commands, run sequentially on the evolving scene. Each
    // must be one undoable unit; `expect_ok` pins accepted vs rejected
    // so a silently-rejected catalog entry fails loudly instead of
    // testing nothing.
    let catalog: &[(&str, bool)] = &[
        ("add_line 30,0 35,2 noconnect", true),
        ("add_point 40,40", true),
        ("add_circle 40,10 2 noconnect", true),
        ("add_arc 30,10 34,10 32,12 noconnect", true),
        ("add_ellipse 40,20 3 1 15 noconnect", true),
        ("add_rect 50,0 55,4", true),
        ("horizontal L4", true),
        ("parallel L4 L5", true),
        ("equal L4 L5", true),
        ("length L4 6", true),
        ("radius A0 2", true),
        ("distance P0 L4 5", true),
        ("param w = 4", true),
        ("quiet L5", true),
        ("constr A1", true),
        ("set_derived d0", true),
        ("set_driven d0", true),
        ("drag L4.p2 19,1", true),
        ("mirror L5 about L4", true),
        ("fillet L6 L7 1", true),
        // Rejections must leave the state untouched.
        ("vertical L4", false),
        ("add_line 1,1 1,1", false),
        ("delete L4 horizontal", true),
        ("delete d1", true),
        ("delete A2", true),
        // Read-only commands add no history and change nothing.
        ("list constraints", true),
        ("info L4", true),
        ("measure L4", true),
        ("dof", true),
    ];

    for &(cmd, expect_ok) in catalog {
        let fp_before = fingerprint(&ctx);
        let params_before = params_of(&mut ctx);
        let groups_before = ctx.history.group_list().len();

        let ok = run_one(&mut ctx, cmd);
        assert_eq!(ok, expect_ok, "acceptance of '{}' changed", cmd);

        let groups_added = ctx.history.group_list().len() - groups_before;
        if !ok || groups_added == 0 {
            // Rejected or read-only: nothing may have changed.
            assert_eq!(fp_before, fingerprint(&ctx), "state changed by '{}'", cmd);
            params_close(&params_before, &params_of(&mut ctx), "no-op", cmd);
            continue;
        }

        let fp_after = fingerprint(&ctx);
        let params_after = params_of(&mut ctx);

        assert!(run_one(&mut ctx, &format!("undo {}", groups_added)), "undo after '{}'", cmd);
        assert_eq!(fp_before, fingerprint(&ctx), "undo of '{}' is not identity", cmd);
        params_close(&params_before, &params_of(&mut ctx), "undo", cmd);

        assert!(run_one(&mut ctx, &format!("redo {}", groups_added)), "redo after '{}'", cmd);
        assert_eq!(fp_after, fingerprint(&ctx), "redo of '{}' is not identity", cmd);
        params_close(&params_after, &params_of(&mut ctx), "redo", cmd);
    }
}
