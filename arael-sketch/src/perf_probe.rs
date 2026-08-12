// Frame-loop timing probes (D2/C4). Ignored by default; run with:
//
//   cargo test -r -p arael-sketch -- --ignored --nocapture perf_probe
//
// Prints per-frame and per-suspect timings on the heavy scenes so the
// perf pass fixes what is measured, not what is suspected.

use std::time::Instant;
use eframe::egui;
use crate::test_harness::{Gui, v};
use crate::coincide::CoincidenceGroups;
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
