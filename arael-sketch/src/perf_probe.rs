// Frame-loop timing probes (D2/C4). Ignored by default; run with:
//
//   cargo test -r -p arael-sketch -- --ignored --nocapture perf_probe
//
// Prints per-frame and per-suspect timings on the heavy scenes so the
// perf pass fixes what is measured, not what is suspected.

use std::time::Instant;
use eframe::egui;
use crate::test_harness::{Gui, v};
use arael_sketch_backend::coincide::CoincidenceGroups;
use arael_sketch_backend::Selection;

/// Median-ish timing: run `f` n times, report (avg_us, max_us).
fn time_us(n: usize, mut f: impl FnMut()) -> (f64, f64) {
    let mut total = 0.0;
    let mut max = 0.0f64;
    for _ in 0..n {
        let t = Instant::now();
        f();
        let us = t.elapsed().as_secs_f64() * 1e6;
        total += us;
        max = max.max(us);
    }
    (total / n as f64, max)
}

fn load_polygon128() -> Gui {
    let mut gui = Gui::new();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/polygon128.cmd");
    let content = std::fs::read_to_string(path).expect("polygon128.cmd");
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for r in gui.app.run_commands(line) {
            assert!(!r.is_error, "'{}' failed: {}", line, r.output);
        }
    }
    // Fit the r=50 polygon into the 1280x800 canvas.
    gui.app.scale = 6.0;
    gui.app.offset = egui::Vec2::new(700.0, 400.0);
    gui.frame();
    gui
}

fn load_robot() -> Option<Gui> {
    // robot.json lives at the repo root as a scratch scene; skip the
    // probe gracefully when absent.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../robot.json");
    let json = std::fs::read_to_string(path).ok()?;
    let mut gui = Gui::new();
    gui.app.load_from_json(&json);
    gui.frame();
    Some(gui)
}

fn probe_scene(name: &str, gui: &mut Gui, drag_from: Option<(f64, f64)>) {
    let s = format!(
        "{}: {} lines, {} arcs, {} points, {} dims",
        name,
        gui.app.sketch.lines.len(),
        gui.app.sketch.arcs.len(),
        gui.app.sketch.points.len(),
        gui.app.sketch.dimensions.len()
    );
    println!("== {}", s);

    // Whole idle frames.
    let (avg, max) = {
        let mut t = (0.0, 0.0f64);
        let n = 100;
        let start = Instant::now();
        let mut worst = 0.0f64;
        for _ in 0..n {
            let f = Instant::now();
            gui.frame();
            worst = worst.max(f.elapsed().as_secs_f64() * 1e6);
        }
        t.0 = start.elapsed().as_secs_f64() * 1e6 / n as f64;
        t.1 = worst;
        t
    };
    println!("  idle frame            avg {:8.1} us   max {:8.1} us", avg, max);

    // Suspects, timed directly.
    let app = &mut gui.app;
    let (avg, max) = time_us(50, || {
        let _ = app.compute_locked_sets();
    });
    println!("  compute_locked_sets   avg {:8.1} us   max {:8.1} us", avg, max);

    let (avg, max) = time_us(50, || {
        let _ = CoincidenceGroups::build(&app.sketch);
    });
    println!("  CoincidenceGroups     avg {:8.1} us   max {:8.1} us", avg, max);

    let sel_a = app.sketch.lines.refs().next().map(Selection::LineP1);
    let sel_b = app.sketch.lines.refs().nth(1).map(Selection::LineP2);
    if let (Some(a), Some(b)) = (sel_a, sel_b) {
        let (avg, max) = time_us(50, || {
            let _ = app.are_transitively_coincident(a, b);
        });
        println!("  transitively_coinc    avg {:8.1} us   max {:8.1} us", avg, max);
    }

    let (avg, max) = time_us(50, || {
        app.build_constraint_markers();
    });
    println!("  build_markers         avg {:8.1} us   max {:8.1} us", avg, max);

    let (avg, max) = time_us(50, || {
        let _ = bincode::serialize(&*app.sketch).unwrap();
    });
    println!("  sketch serialize      avg {:8.1} us   max {:8.1} us", avg, max);

    let snap = bincode::serialize(&*app.sketch).unwrap();
    let (avg, max) = time_us(50, || {
        let _: arael_sketch_solver::Sketch = bincode::deserialize(&snap).unwrap();
    });
    println!("  sketch deserialize    avg {:8.1} us   max {:8.1} us", avg, max);

    let (avg, max) = time_us(20, || {
        let _ = app.sketch.solve();
    });
    println!("  solve                 avg {:8.1} us   max {:8.1} us", avg, max);

    // Drag frames: press on a vertex and orbit the mouse.
    if let Some((x, y)) = drag_from {
        gui.move_to(v(x, y));
        gui.press(v(x, y));
        // Get past the drag threshold.
        gui.move_to(v(x + 0.5, y + 0.5));
        let n = 60;
        let start = Instant::now();
        let mut worst = 0.0f64;
        for i in 0..n {
            let ang = i as f64 / n as f64 * std::f64::consts::TAU;
            let p = v(x + 2.0 * ang.cos(), y + 2.0 * ang.sin());
            let f = Instant::now();
            gui.move_to(p);
            worst = worst.max(f.elapsed().as_secs_f64() * 1e6);
        }
        let avg = start.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!("  drag frame            avg {:8.1} us   max {:8.1} us", avg, worst);
        gui.release(v(x, y));
    }
}

#[test]
#[ignore = "timing probe, run manually with --ignored --nocapture in release"]
fn perf_probe_polygon128() {
    let mut gui = load_polygon128();
    probe_scene("polygon128", &mut gui, Some((50.0, 0.0)));
}

/// Decompose the drag-frame cost: time update_drag and its pieces
/// individually during a live drag on the polygon vertex.
#[test]
#[ignore = "timing probe, run manually with --ignored --nocapture in release"]
fn perf_probe_polygon128_drag_pieces() {
    let mut gui = load_polygon128();
    let (x, y) = (50.0, 0.0);
    gui.move_to(v(x, y));
    gui.press(v(x, y));
    gui.move_to(v(x + 2.0, y + 2.0));
    assert!(gui.app.grab.is_some(), "drag should be active");
    let threshold = 15.0 / gui.app.scale as f64;
    println!("== polygon128 drag pieces");

    let app = &mut gui.app;
    let helper = app.drag_apparatus.as_ref().unwrap().helper;

    // Whole update_drag at orbiting positions.
    let mut i = 0usize;
    let (avg, max) = time_us(60, || {
        let ang = i as f64 / 60.0 * std::f64::consts::TAU;
        i += 1;
        let p = v(x + 2.0 * ang.cos(), y + 2.0 * ang.sin());
        app.update_drag(p, threshold);
    });
    println!("  update_drag           avg {:8.1} us   max {:8.1} us", avg, max);

    // Piece: move helper + solve (the physics half).
    let mut i = 0usize;
    let (avg, max) = time_us(60, || {
        let ang = i as f64 / 60.0 * std::f64::consts::TAU;
        i += 1;
        let p = v(x + 2.0 * ang.cos(), y + 2.0 * ang.sin());
        app.sketch.mutate_values(|s| s.move_drag_helper(helper, p));
        let _ = app.sketch.solve();
    });
    println!("  move + solve          avg {:8.1} us   max {:8.1} us", avg, max);

    // Piece: snap-target scan.
    let mut i = 0usize;
    let (avg, max) = time_us(60, || {
        let ang = i as f64 / 60.0 * std::f64::consts::TAU;
        i += 1;
        let p = v(x + 2.0 * ang.cos(), y + 2.0 * ang.sin());
        let _ = app.find_snap_target(p, threshold);
    });
    println!("  find_snap_target      avg {:8.1} us   max {:8.1} us", avg, max);

    // Piece: the best-cost snapshot chain update_drag runs per good
    // frame (serialize live, deserialize clone, strip apparatus,
    // serialize clean).
    let (avg, max) = {
        let app: &crate::EditorApp = app;
        time_us(60, || {
            let snap = bincode::serialize(&*app.sketch).unwrap();
            let mut clean: arael_sketch_solver::Sketch = bincode::deserialize(&snap).unwrap();
            if let Some(a) = &app.drag_apparatus {
                clean.remove_drag(a);
            }
            let _ = bincode::serialize(&clean).unwrap();
        })
    };
    println!("  snapshot chain        avg {:8.1} us   max {:8.1} us", avg, max);

    // Pieces of the perp-hint machinery (line-endpoint drags only).
    if let Some(crate::tools::GrabTarget::LineP1(line) | crate::tools::GrabTarget::LineP2(line)) = app.grab {
        let is_p1 = matches!(app.grab, Some(crate::tools::GrabTarget::LineP1(_)));
        let (avg, max) = time_us(60, || {
            let _ = app.find_anchor_host_line_for_drag(line, is_p1);
        });
        println!("  find_anchor_host      avg {:8.1} us   max {:8.1} us", avg, max);
        if let Some(host) = app.find_anchor_host_line_for_drag(line, is_p1) {
            let (avg, max) = time_us(60, || {
                let _ = arael_sketch_backend::conflicts::validate_action(
                    &app.sketch,
                    &arael_sketch_backend::Action::ApplyPerpendicular { a: line, b: host });
            });
            println!("  validate(perp)        avg {:8.1} us   max {:8.1} us", avg, max);
            let (avg, max) = time_us(60, || {
                let _ = app.perp_would_reduce_dof(line, host);
            });
            println!("  perp_reduce_dof       avg {:8.1} us   max {:8.1} us", avg, max);
        }
        let anchor = if is_p1 { app.sketch.lines[line].p2.value } else { app.sketch.lines[line].p1.value };
        let (avg, max) = time_us(60, || {
            let _ = app.find_best_collinear_host_at(anchor, v(x + 1.0, y + 1.0), crate::PERP_SNAP_PX, Some(line));
        });
        println!("  collinear_host        avg {:8.1} us   max {:8.1} us", avg, max);
    }

    gui.release(v(x, y));
}

#[test]
#[ignore = "timing probe, run manually with --ignored --nocapture in release"]
fn perf_probe_robot() {
    let Some(mut gui) = load_robot() else {
        println!("robot.json not present, skipping");
        return;
    };
    probe_scene("robot", &mut gui, None);
}

/// 11x11 symmetric rect pattern on the startup sketch, command path.
#[test]
#[ignore]
fn perf_probe_pattern11() {
    let mut gui = Gui::new();
    // Swap in the startup sketch (EditorApp::default keeps the demo scene).
    gui.app = crate::EditorApp::default();
    gui.frame();

    let t = Instant::now();
    for r in gui.app.run_commands("select all") {
        assert!(!r.is_error, "{}", r.output);
    }
    println!("select all          {:9.1} us", t.elapsed().as_secs_f64() * 1e6);

    let t = Instant::now();
    for r in gui.app.run_commands("pattern rect selection 11 9 symmetric by 11 9 symmetric") {
        assert!(!r.is_error, "{}", r.output);
    }
    println!("pattern cmd (gui)   {:9.1} us", t.elapsed().as_secs_f64() * 1e6);

    for i in 0..10 {
        let t = Instant::now();
        gui.frame();
        println!("frame {:2}            {:9.1} us", i, t.elapsed().as_secs_f64() * 1e6);
    }
    let (avg, max) = time_us(30, || {
        let _ = CoincidenceGroups::build(&gui.app.sketch);
    });
    println!("CoincidenceGroups   avg {:9.1} us  max {:9.1} us", avg, max);
}

/// The same 11x11 pattern via the GUI tool flow (plan, apply, frames).
#[test]
#[ignore]
fn perf_probe_pattern11_tool() {
    arael_sketch_solver::set_verbose(true);
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();

    for r in gui.app.run_commands("select all") {
        assert!(!r.is_error, "{}", r.output);
    }

    let t = Instant::now();
    gui.app.enter_pattern_tool();
    println!("enter tool          {:9.1} us", t.elapsed().as_secs_f64() * 1e6);

    {
        let st = gui.app.pattern_tool.as_mut().unwrap();
        st.kind = crate::pattern_tool::PatternToolKind::Rectangular;
        st.quantity1 = "11".into();
        st.distance1 = "9".into();
        st.symmetric1 = true;
        st.quantity2 = "11".into();
        st.distance2 = "9".into();
        st.symmetric2 = true;
    }
    let t = Instant::now();
    gui.app.refresh_pattern_plan();
    println!("refresh plan        {:9.1} us", t.elapsed().as_secs_f64() * 1e6);

    for i in 0..5 {
        let t = Instant::now();
        gui.frame();
        println!("tool frame {:2}       {:9.1} us", i, t.elapsed().as_secs_f64() * 1e6);
    }

    let t = Instant::now();
    gui.app.apply_pattern();
    println!("apply_pattern       {:9.1} us", t.elapsed().as_secs_f64() * 1e6);

    for i in 0..10 {
        let t = Instant::now();
        gui.frame();
        println!("post frame {:2}       {:9.1} us", i, t.elapsed().as_secs_f64() * 1e6);
    }
}


/// Grid pattern on the startup sketch, select all, Backspace in the GUI.
fn probe_delete_grid(n: u32) {
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();
    for r in gui.app.run_commands(&format!(
        "select all; pattern rect selection {} 9 symmetric by {} 9 symmetric", n, n)) {
        assert!(!r.is_error, "{}", r.output);
    }
    for r in gui.app.run_commands("select all") {
        assert!(!r.is_error, "{}", r.output);
    }
    let sel = gui.app.selection.len();
    let t = Instant::now();
    gui.key(egui::Key::Backspace);
    println!("delete {:2}x{:<2} ({} selected)  {:9.1} ms", n, n, sel,
        t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(gui.line_count(), 0);
    assert_eq!(gui.arc_count(), 0);
}

#[test]
#[ignore]
fn perf_probe_delete_grid5() { probe_delete_grid(5); }

#[test]
#[ignore]
fn perf_probe_delete_grid7() { probe_delete_grid(7); }

#[test]
#[ignore]
fn perf_probe_delete_grid15() { probe_delete_grid(15); }

/// Grid pattern, every vertex selected, Lock applied over the selection.
fn probe_lock_grid(n: u32) {
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();
    for r in gui.app.run_commands(&format!(
        "select all; pattern rect selection {} 9 symmetric by {} 9 symmetric", n, n)) {
        assert!(!r.is_error, "{}", r.output);
    }
    gui.app.selection.clear();
    let mut sel = Vec::new();
    for r in gui.app.sketch.lines.refs() {
        sel.push(Selection::LineP1(r));
        sel.push(Selection::LineP2(r));
    }
    for r in gui.app.sketch.arcs.refs() {
        sel.push(Selection::ArcCenter(r));
    }
    gui.app.selection = sel;
    let n_sel = gui.app.selection.len();
    let t = Instant::now();
    gui.app.apply_lock();
    println!("lock {:2}x{:<2} ({} selected)    {:9.1} ms", n, n, n_sel,
        t.elapsed().as_secs_f64() * 1e3);
    gui.frame();
}

/// Grid pattern, select all, construction toggled over the selection.
fn probe_constr_grid(n: u32) {
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();
    for r in gui.app.run_commands(&format!(
        "select all; pattern rect selection {} 9 symmetric by {} 9 symmetric; select all", n, n)) {
        assert!(!r.is_error, "{}", r.output);
    }
    let n_sel = gui.app.selection.len();
    let t = Instant::now();
    gui.app.apply_toggle_construction();
    println!("constr {:2}x{:<2} ({} selected)  {:9.1} ms", n, n, n_sel,
        t.elapsed().as_secs_f64() * 1e3);
    gui.frame();
}

#[test]
#[ignore]
fn perf_probe_lock_grid7() { probe_lock_grid(7); }

#[test]
#[ignore]
fn perf_probe_lock_grid15() { probe_lock_grid(15); }

#[test]
#[ignore]
fn perf_probe_constr_grid7() { probe_constr_grid(7); }

#[test]
#[ignore]
fn perf_probe_constr_grid15() { probe_constr_grid(15); }

/// Grid pattern plus an axis, mirror the whole selection about it.
#[test]
#[ignore]
fn perf_probe_mirror_grid9() {
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();
    for r in gui.app.run_commands(
        "select all; pattern rect selection 9 9 symmetric by 9 9 symmetric") {
        assert!(!r.is_error, "{}", r.output);
    }
    for r in gui.app.run_commands("add_line 60,-60 60,60 noconnect nocursor; select all") {
        assert!(!r.is_error, "{}", r.output);
    }
    let n_sel = gui.app.selection.len();
    let t = Instant::now();
    for r in gui.app.run_commands("mirror selection about L243") {
        assert!(!r.is_error, "{}", r.output);
    }
    println!("mirror 9x9 ({} selected)   {:9.1} ms", n_sel,
        t.elapsed().as_secs_f64() * 1e3);
    gui.frame();
}

/// Temporary probe: dense vs iterative rank cost at the pathological
/// intermediate shape (m=3642, n=1936, nullity ~1803).
#[test]
#[ignore]
fn perf_probe_rank_shape() {
    use arael::model::{Jacobian, JacobianRow};
    let n = 1936usize;
    let constrained = 133usize; // params actually touched -> nullity 1803
    let mut rng = 0x12345678u64;
    let mut next = move || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut rows = Vec::new();
    for i in 0..3642usize {
        let a = (i * 7) % constrained;
        let b = (i * 13 + 5) % constrained;
        rows.push(JacobianRow {
            constraint: i as u32,
            label: "syn",
            residual: 0.0,
            entries: vec![(a as u32, next()), (b as u32, next()), ((a + 1) as u32 % constrained as u32, next()), ((b + 3) as u32 % constrained as u32, next())],
        });
    }
    let jac = Jacobian { num_params: n, rows };

    let opts_dense = arael::rank::RankOptions { dense_cutoff: usize::MAX, ..Default::default() };
    let t = Instant::now();
    let r = jac.numeric_rank(&opts_dense).unwrap();
    println!("cutoff=MAX rank={} nullity={} method={:?} in {:.3}s", r.rank, r.nullity, r.method, t.elapsed().as_secs_f64());

    let opts_iter = arael::rank::RankOptions { null_hint: Some(3), ..Default::default() };
    let t = Instant::now();
    let r = jac.numeric_rank(&opts_iter).unwrap();
    println!("cutoff=0   rank={} nullity={} method={:?} in {:.3}s", r.rank, r.nullity, r.method, t.elapsed().as_secs_f64());
}

/// Loose scene: many free doodle lines around one constrained part.
#[test]
#[ignore]
fn perf_probe_rank_loose_scene() {
    let mut gui = Gui::new();
    gui.app = crate::EditorApp::default();
    gui.frame();
    for i in 0..200 {
        let y = i as f64 * 0.5;
        for r in gui.app.run_commands(&format!(
            "add_line {},{} {},{} noconnect nocursor", 20.0 + y, y, 25.0 + y, y + 3.0)) {
            assert!(!r.is_error, "{}", r.output);
        }
    }
    let (avg, max) = time_us(10, || {
        gui.app.sketch.mutate_values(|s| {
            s.clear_cached_dof();
            let _ = s.ensure_rank();
        });
    });
    println!("rank, 200 free lines + part   avg {:9.1} us  max {:9.1} us", avg, max);
}

/// Large-scene profile: DOF, add_line between endpoints, drag steps,
/// and the solve's assembly vs linear-solve split. Loads
/// ~/large_grid.json; skips gracefully when absent.
#[test]
#[ignore]
fn perf_probe_large_grid() {
    let path = format!("{}/large_grid.json", std::env::var("HOME").unwrap_or_default());
    let Ok(json) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: {} not found", path);
        return;
    };
    arael_sketch_solver::set_verbose(true);
    let mut gui = Gui::new();
    gui.app.load_from_json(&json);
    gui.frame();
    let s = &gui.app.sketch;
    println!("scene: {} lines, {} points, {} arcs",
        s.lines.refs().count(), s.points.refs().count(), s.arcs.refs().count());

    // 1. DOF (cold: cache cleared each time).
    for i in 0..3 {
        let t = Instant::now();
        gui.app.sketch.mutate_values(|s| {
            s.clear_cached_dof();
            let _ = s.ensure_rank();
        });
        println!("dof cold {}          {:9.1} ms", i, t.elapsed().as_secs_f64() * 1e3);
    }

    // Building blocks, standalone.
    let (avg, max) = time_us(5, || {
        let _ = bincode::serialize(&*gui.app.sketch).unwrap();
    });
    println!("bincode snapshot     avg {:9.1} ms  max {:9.1} ms", avg / 1e3, max / 1e3);
    let (avg, max) = time_us(5, || {
        let _ = gui.app.sketch.current_cost();
    });
    println!("current_cost         avg {:9.1} ms  max {:9.1} ms", avg / 1e3, max / 1e3);

    // 2. add_line between two existing endpoints (auto-connect snaps).
    let lines: Vec<_> = gui.app.sketch.lines.refs().collect();
    for (i, (a, b)) in [(lines[7], lines[lines.len() / 2]), (lines[13], lines[lines.len() - 3])]
        .iter().enumerate()
    {
        let (p1, p2) = {
            let s = &gui.app.sketch;
            (s.lines[*a].p1.value, s.lines[*b].p2.value)
        };
        let t = Instant::now();
        for r in gui.app.run_commands(&format!("add_line {},{} {},{}", p1.x, p1.y, p2.x, p2.y)) {
            assert!(!r.is_error, "{}", r.output);
        }
        println!("add_line {}           {:9.1} ms", i, t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        for r in gui.app.run_commands("undo") {
            assert!(!r.is_error, "{}", r.output);
        }
        println!("undo                 {:9.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }

    // 3. drag steps + the per-frame suspects, via the shared scene probe.
    let target = {
        let s = &gui.app.sketch;
        s.lines[lines[lines.len() / 3]].p1.value
    };
    gui.app.scale = 40.0;
    gui.app.offset = egui::Vec2::new(
        400.0 - (target.x * 40.0) as f32,
        300.0 + (target.y * 40.0) as f32,
    );
    gui.frame();
    probe_scene("large_grid", &mut gui, Some((target.x, target.y)));

    // 4. solve: assembly vs optimization. Displace a point, solve with
    // timing gathered (verbose wires gather_timing).
    gui.app.sketch.mutate_values(|s| {
        let r = s.lines.refs().next().unwrap();
        s.lines[r].p1.value.x += 3.0;
        s.lines[r].p1.value.y += 2.0;
    });
    let t = Instant::now();
    let result = gui.app.sketch.solve();
    let total = t.elapsed().as_secs_f64() * 1e3;
    println!("solve: {:.1} ms total, {} iters ({} accepted), cost {:.3e} -> {:.3e}",
        total, result.iterations, result.accepted_iterations, result.start_cost, result.end_cost);
    if let Some(tm) = &result.timing {
        println!("  assembly       {:9.1} ms ({} calls, first {:.1} ms)",
            tm.assembly.as_secs_f64() * 1e3, tm.assembly_count,
            tm.first_assembly.as_secs_f64() * 1e3);
        println!("  analysis       {:9.1} ms", tm.analysis.as_secs_f64() * 1e3);
        println!("  linear_solve   {:9.1} ms (first {:.1} ms)",
            tm.linear_solve.as_secs_f64() * 1e3, tm.first_linear_solve.as_secs_f64() * 1e3);
        println!("  cost_eval      {:9.1} ms", tm.cost_eval.as_secs_f64() * 1e3);
        println!("  advance        {:9.1} ms", tm.advance.as_secs_f64() * 1e3);
        println!("  lm total       {:9.1} ms", tm.total.as_secs_f64() * 1e3);
    } else {
        println!("  (no timing gathered)");
    }
}
