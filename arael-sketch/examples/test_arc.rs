use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch::*;

fn main() {
    let mut sketch = Sketch::new();

    // Create default triangle (same as editor default)
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(3.0, 0.0), vect2d::new(1.5, 2.5));
    let l2 = sketch.add_line(vect2d::new(1.5, 2.5), vect2d::new(0.0, 0.0));
    sketch.coincident_ll21.push(CoincidentLL21 { a: l0, b: l1, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l1, b: l2, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l0, hb: CrossBlock::new() });
    sketch.solve();

    eprintln!("=== Before arc ===");
    eprintln!("L0: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
        sketch.lines[l0].p1.value.x, sketch.lines[l0].p1.value.y,
        sketch.lines[l0].p2.value.x, sketch.lines[l0].p2.value.y);
    eprintln!("L1: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
        sketch.lines[l1].p1.value.x, sketch.lines[l1].p1.value.y,
        sketch.lines[l1].p2.value.x, sketch.lines[l1].p2.value.y);
    eprintln!("L2: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
        sketch.lines[l2].p1.value.x, sketch.lines[l2].p1.value.y,
        sketch.lines[l2].p2.value.x, sketch.lines[l2].p2.value.y);

    // Arc from L1.p2 to L0.p2 with mid point outside (below triangle)
    let start = sketch.lines[l1].p2.value;  // should be ~(1.5, 2.5)
    let end = sketch.lines[l0].p2.value;    // should be ~(3.0, 0.0)
    let mid = vect2d::new(3.5, 1.5);        // outside, to the right

    eprintln!("\nArc start=({:.2},{:.2}) end=({:.2},{:.2}) mid=({:.2},{:.2})", 
        start.x, start.y, end.x, end.y, mid.x, mid.y);

    if let Some((c, r, sa, ea, swapped)) = circumscribed_arc(start, end, mid) {
        eprintln!("Circumscribed: center=({:.2},{:.2}) r={:.2} sa={:.4} ea={:.4} swapped={}", c.x, c.y, r, sa, ea, swapped);

        let arc = sketch.add_arc(c, r, sa, ea, false);
        sketch.solve();

        eprintln!("\n=== After add_arc + solve ===");
        let a = &sketch.arcs[arc];
        eprintln!("Arc: center=({:.2},{:.2}) r={:.2} sa={:.4} ea={:.4}",
            a.center.value.x, a.center.value.y, a.radius.value, a.start_angle.value, a.end_angle.value);
        let arc_s = vect2d::new(a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                                 a.center.value.y + a.radius.value * a.start_angle.value.sin());
        let arc_e = vect2d::new(a.center.value.x + a.radius.value * a.end_angle.value.cos(),
                                 a.center.value.y + a.radius.value * a.end_angle.value.sin());
        eprintln!("Arc start pos=({:.2},{:.2}) end pos=({:.2},{:.2})", arc_s.x, arc_s.y, arc_e.x, arc_e.y);

        eprintln!("L0: p1=({:.2},{:.2}) p2=({:.2},{:.2})",
            sketch.lines[l0].p1.value.x, sketch.lines[l0].p1.value.y,
            sketch.lines[l0].p2.value.x, sketch.lines[l0].p2.value.y);
        eprintln!("L1: p1=({:.2},{:.2}) p2=({:.2},{:.2})",
            sketch.lines[l1].p1.value.x, sketch.lines[l1].p1.value.y,
            sketch.lines[l1].p2.value.x, sketch.lines[l1].p2.value.y);

        // Use swapped flag: when swapped, arc.start_angle = end click, arc.end_angle = start click
        // start click = L1.p2, end click = L0.p2
        if swapped {
            eprintln!("\nSwapped! Arc start_angle = end click (L0.p2), end_angle = start click (L1.p2)");
            // L1.p2 should match arc END
            sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd { line: l1, arc, hb: CrossBlock::new() });
        } else {
            // L1.p2 should match arc START
            sketch.coincident_lp2_arc_start.push(CoincidentLP2ArcStart { line: l1, arc, hb: CrossBlock::new() });
        }
        sketch.solve();

        eprintln!("\n=== After CoincidentLP2ArcStart (L1.p2 = arc.start) ===");
        let a = &sketch.arcs[arc];
        eprintln!("Arc: center=({:.2},{:.2}) r={:.2} sa={:.4} ea={:.4}", 
            a.center.value.x, a.center.value.y, a.radius.value, a.start_angle.value, a.end_angle.value);
        let arc_s = vect2d::new(a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                                 a.center.value.y + a.radius.value * a.start_angle.value.sin());
        eprintln!("Arc start pos=({:.2},{:.2})", arc_s.x, arc_s.y);
        eprintln!("L0: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
            sketch.lines[l0].p1.value.x, sketch.lines[l0].p1.value.y,
            sketch.lines[l0].p2.value.x, sketch.lines[l0].p2.value.y);
        eprintln!("L1: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
            sketch.lines[l1].p1.value.x, sketch.lines[l1].p1.value.y,
            sketch.lines[l1].p2.value.x, sketch.lines[l1].p2.value.y);

        // L0.p2 should match the OTHER arc endpoint
        if swapped {
            sketch.coincident_lp2_arc_start.push(CoincidentLP2ArcStart { line: l0, arc, hb: CrossBlock::new() });
        } else {
            sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd { line: l0, arc, hb: CrossBlock::new() });
        }
        sketch.solve();

        eprintln!("\n=== After second coincident ===");
        let a = &sketch.arcs[arc];
        eprintln!("Arc: center=({:.2},{:.2}) r={:.2} sa={:.4} ea={:.4}", 
            a.center.value.x, a.center.value.y, a.radius.value, a.start_angle.value, a.end_angle.value);
        let arc_s = vect2d::new(a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                                 a.center.value.y + a.radius.value * a.start_angle.value.sin());
        let arc_e = vect2d::new(a.center.value.x + a.radius.value * a.end_angle.value.cos(),
                                 a.center.value.y + a.radius.value * a.end_angle.value.sin());
        eprintln!("Arc start pos=({:.2},{:.2}) end pos=({:.2},{:.2})", arc_s.x, arc_s.y, arc_e.x, arc_e.y);
        eprintln!("L0: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
            sketch.lines[l0].p1.value.x, sketch.lines[l0].p1.value.y,
            sketch.lines[l0].p2.value.x, sketch.lines[l0].p2.value.y);
        eprintln!("L1: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
            sketch.lines[l1].p1.value.x, sketch.lines[l1].p1.value.y,
            sketch.lines[l1].p2.value.x, sketch.lines[l1].p2.value.y);
        eprintln!("L2: p1=({:.2},{:.2}) p2=({:.2},{:.2})", 
            sketch.lines[l2].p1.value.x, sketch.lines[l2].p1.value.y,
            sketch.lines[l2].p2.value.x, sketch.lines[l2].p2.value.y);
    }
}

// Copy circumscribed_arc from editor
fn circumscribed_arc(p1: vect2d, p2: vect2d, p3: vect2d) -> Option<(vect2d, f64, f64, f64, bool)> {
    let ax = p1.x; let ay = p1.y;
    let bx = p2.x; let by = p2.y;
    let cx = p3.x; let cy = p3.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 { return None; }
    let aa = ax * ax + ay * ay;
    let bb = bx * bx + by * by;
    let cc = cx * cx + cy * cy;
    let ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d;
    let uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d;
    let center = vect2d::new(ux, uy);
    let radius = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();
    let sa = (ay - uy).atan2(ax - ux);
    let ea = (by - uy).atan2(bx - ux);
    let ma = (cy - uy).atan2(cx - ux);
    let norm = |a: f64| -> f64 { let r = a % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
    let span_ccw = norm(ea - sa);
    let mid_ccw = norm(ma - sa);
    if mid_ccw < span_ccw {
        Some((center, radius, sa, ea, false))
    } else {
        Some((center, radius, ea, sa, true))
    }
}
