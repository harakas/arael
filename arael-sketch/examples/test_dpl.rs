use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch::*;

fn main() {
    let mut sketch = Sketch::new();
    
    // Horizontal line from (0,0) to (4,0)
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0));
    // Point at (2, 1) - 1 unit above the line
    let p = sketch.add_point(vect2d::new(2.0, 1.0));
    
    sketch.solve();
    eprintln!("Before constraint:");
    eprintln!("  Line: ({:.2},{:.2})->({:.2},{:.2})", 
        sketch.lines[l].p1.value.x, sketch.lines[l].p1.value.y,
        sketch.lines[l].p2.value.x, sketch.lines[l].p2.value.y);
    eprintln!("  Point: ({:.2},{:.2})", sketch.points[p].pos.value.x, sketch.points[p].pos.value.y);
    
    // Measure signed distance
    let lv = &sketch.lines[l];
    let dx = lv.p2.value.x - lv.p1.value.x;
    let dy = lv.p2.value.y - lv.p1.value.y;
    let len = (dx*dx + dy*dy).sqrt();
    let signed = ((sketch.points[p].pos.value.x - lv.p1.value.x) * dy 
                - (sketch.points[p].pos.value.y - lv.p1.value.y) * dx) / len;
    eprintln!("  Signed dist = {:.4}", signed);
    
    // Constrain distance to 2.0 (keeping sign)
    let constraint_dist = if signed >= 0.0 { 2.0 } else { -2.0 };
    sketch.distance_pl.push(DistancePL {
        point: p, line: l, distance: constraint_dist, hb: CrossBlock::new(),
    });
    sketch.solve();
    
    eprintln!("\nAfter constraint (target dist=2.0, signed={:.1}):", constraint_dist);
    eprintln!("  Line: ({:.2},{:.2})->({:.2},{:.2})", 
        sketch.lines[l].p1.value.x, sketch.lines[l].p1.value.y,
        sketch.lines[l].p2.value.x, sketch.lines[l].p2.value.y);
    eprintln!("  Point: ({:.2},{:.2})", sketch.points[p].pos.value.x, sketch.points[p].pos.value.y);
    
    // Verify
    let lv = &sketch.lines[l];
    let dx = lv.p2.value.x - lv.p1.value.x;
    let dy = lv.p2.value.y - lv.p1.value.y;
    let len = (dx*dx + dy*dy).sqrt();
    let actual = ((sketch.points[p].pos.value.x - lv.p1.value.x) * dy 
                - (sketch.points[p].pos.value.y - lv.p1.value.y) * dx) / len;
    eprintln!("  Actual signed dist = {:.4}", actual);
    eprintln!("  Actual abs dist = {:.4}", actual.abs());
}
