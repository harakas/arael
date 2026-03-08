use arael::refs::{Ref, Vec, Deque, Arena};
use arael::info;

// A simple pose with position and orientation
#[derive(Debug)]
#[allow(dead_code)]
struct Pose {
    x: f64,
    y: f64,
    heading: f64,
}

// A landmark observed from a pose
#[derive(Debug)]
struct Observation {
    pose: Ref<Pose>,
    range: f64,
    bearing: f64,
}

// A landmark in the map
#[derive(Debug)]
struct Landmark {
    x: f64,
    y: f64,
    observations: Vec<Observation>,
}

fn main() {
    // ==========================================================
    // Vec basics — push returns a Ref, use it to index back
    // ==========================================================
    info!("=== Vec basics ===");
    info!("");

    let mut landmarks: Vec<Landmark> = Vec::new();
    let lm0 = landmarks.push(Landmark { x: 5.0, y: 3.0, observations: Vec::new() });
    let lm1 = landmarks.push(Landmark { x: -2.0, y: 7.0, observations: Vec::new() });

    info!("Landmark 0: ({}, {})", landmarks[lm0].x, landmarks[lm0].y);
    info!("Landmark 1: ({}, {})", landmarks[lm1].x, landmarks[lm1].y);

    // Refs are typed — Ref<Landmark> can't accidentally index a Vec<Pose>
    // (this would be a compile error):
    //   let poses: Vec<Pose> = Vec::new();
    //   poses[lm0];  // ERROR: expected Ref<Pose>, got Ref<Landmark>

    // ==========================================================
    // Deque — a sliding window of poses with stable references
    // ==========================================================
    info!("");
    info!("=== Deque — sliding window ===");
    info!("");

    let mut poses: Deque<Pose> = Deque::new();

    // Robot moves and records poses
    let p0 = poses.push_back(Pose { x: 0.0, y: 0.0, heading: 0.0 });
    let p1 = poses.push_back(Pose { x: 1.0, y: 0.0, heading: 0.1 });
    let p2 = poses.push_back(Pose { x: 2.0, y: 0.5, heading: 0.3 });
    let p3 = poses.push_back(Pose { x: 3.0, y: 1.0, heading: 0.5 });

    info!("Poses in window: {}", poses.len());
    info!("Pose p2: {:?}", poses[p2]);

    // Landmarks store Refs to poses where they were observed
    landmarks[lm0].observations.push(Observation { pose: p1, range: 4.5, bearing: 0.6 });
    landmarks[lm0].observations.push(Observation { pose: p3, range: 2.1, bearing: -0.2 });
    landmarks[lm1].observations.push(Observation { pose: p2, range: 6.0, bearing: 1.1 });

    info!("");
    info!("Landmark 0 observed from {} poses", landmarks[lm0].observations.len());
    for obs in landmarks[lm0].observations.iter() {
        info!("  from pose {:?}: range={:.1}, bearing={:.2}", obs.pose, obs.range, obs.bearing);
        info!("    pose position: ({}, {})", poses[obs.pose].x, poses[obs.pose].y);
    }

    // ==========================================================
    // Evicting old poses — refs go stale
    // ==========================================================
    info!("");
    info!("=== Stale refs ===");
    info!("");

    // Drop the oldest poses to keep a sliding window
    info!("Before eviction: {} poses, front={:?} back={:?}",
        poses.len(), poses.front_ref(), poses.back_ref());

    poses.pop_front(); // evict p0
    poses.pop_front(); // evict p1

    info!("After eviction:  {} poses, front={:?} back={:?}",
        poses.len(), poses.front_ref(), poses.back_ref());

    info!("");
    // p2 and p3 still work fine
    info!("p2 still valid: contains_ref={}, value={:?}", poses.contains_ref(p2), poses[p2]);
    info!("p3 still valid: contains_ref={}, value={:?}", poses.contains_ref(p3), poses[p3]);

    info!("");
    // p0 and p1 are now stale!
    info!("p0 is stale: contains_ref={}", poses.contains_ref(p0));
    info!("p1 is stale: contains_ref={}", poses.contains_ref(p1));

    // Safe access with get() returns None for stale refs
    info!("");
    info!("get(p0) = {:?}", poses.get(p0));
    info!("get(p1) = {:?}", poses.get(p1));
    info!("get(p2) = {:?}", poses.get(p2));

    // In practice: check observations before using them
    info!("");
    info!("=== Filtering stale observations ===");
    info!("");

    let obs = &landmarks[lm0].observations;
    for (i, o) in obs.iter().enumerate() {
        if poses.contains_ref(o.pose) {
            let p = &poses[o.pose];
            info!("  obs[{}]: pose ({}, {}) range={:.1}", i, p.x, p.y, o.range);
        } else {
            info!("  obs[{}]: STALE (pose {:?} was evicted)", i, o.pose);
        }
    }

    // ==========================================================
    // Arena — stable refs even after deletion
    // ==========================================================
    info!("");
    info!("=== Arena — deletion with stable refs ===");
    info!("");

    let mut items: Arena<&str> = Arena::new();
    let a = items.push("alpha");
    let b = items.push("beta");
    let c = items.push("gamma");
    let d = items.push("delta");

    info!("Before deletion: {} items", items.len());
    for r in items.refs() {
        info!("  {:?} -> {}", r, items[r]);
    }

    // Remove from the middle
    info!("");
    info!("Removing {:?} ({})", b, items[b]);
    items.remove(b);

    info!("After deletion: {} items", items.len());
    for r in items.refs() {
        info!("  {:?} -> {}", r, items[r]);
    }

    // Other refs are still valid
    info!("");
    info!("a={}, c={}, d={}", items[a], items[c], items[d]);
    info!("contains(b) = {}", items.contains(b));
    info!("get(b) = {:?}", items.get(b));
}
