// Headless GUI test harness: drives EditorApp::update_app through
// egui::Context::run with synthetic input, no window, no GPU, no
// pixels. Tests speak sketch coordinates and assert on app/sketch
// state. See gui_tests.rs for the suite.

use eframe::egui;
use arael::vect::vect2d;
use arael_sketch_backend::Selection;
use crate::EditorApp;

/// Fixed headless window size. Sketch coords map through the app's
/// default transform (offset 400,300 / scale 80), so sketch x in
/// roughly -3..10 and y in -6..3 lands inside the canvas.
const SCREEN: egui::Vec2 = egui::Vec2::new(1280.0, 800.0);

/// Frames per simulated second; each frame() advances time by this.
const DT: f64 = 1.0 / 60.0;

pub struct Gui {
    pub app: EditorApp,
    pub ctx: egui::Context,
    time: f64,
    pub modifiers: egui::Modifiers,
    events: Vec<egui::Event>,
}

impl Gui {
    /// App over an empty sketch (same construction as `--empty`),
    /// with one warmup frame run so panels exist.
    pub fn new() -> Self {
        let mut app = EditorApp::default();
        let empty = serde_json::to_string(&arael_sketch_solver::Sketch::new()).unwrap();
        app.load_from_json(&empty);
        let mut gui = Gui {
            app,
            ctx: egui::Context::default(),
            time: 0.0,
            modifiers: egui::Modifiers::default(),
            events: Vec::new(),
        };
        gui.frame();
        gui
    }

    /// Run a backend command for scene setup (arrange); gestures are
    /// the thing under test (act).
    pub fn cmd(&mut self, s: &str) {
        for r in self.app.run_commands(s) {
            assert!(!r.is_error, "setup command '{}' failed: {}", s, r.output);
        }
        self.frame();
    }

    /// Run one frame with the queued events, then check invariants.
    pub fn frame(&mut self) {
        self.time += DT;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
            time: Some(self.time),
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            ..Default::default()
        };
        let ctx = self.ctx.clone();
        let _ = ctx.run(input, |c| self.app.update_app(c));
        self.check_invariants();
    }

    pub fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.frame();
        }
    }

    pub fn screen(&self, p: vect2d) -> egui::Pos2 {
        let sp = self.app.to_screen(p);
        assert!(
            sp.x > 130.0 && sp.x < SCREEN.x && sp.y > 0.0 && sp.y < SCREEN.y,
            "sketch point ({}, {}) maps to screen ({}, {}) outside the canvas",
            p.x, p.y, sp.x, sp.y
        );
        sp
    }

    pub fn move_to(&mut self, p: vect2d) {
        let sp = self.screen(p);
        self.events.push(egui::Event::PointerMoved(sp));
        self.frame();
    }

    fn button(&mut self, p: vect2d, pressed: bool) {
        let sp = self.screen(p);
        self.events.push(egui::Event::PointerButton {
            pos: sp,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: self.modifiers,
        });
        self.frame();
    }

    /// Full primary-button click at a sketch position. Advances time
    /// past egui's double-click window afterwards so two consecutive
    /// click() calls stay independent clicks.
    pub fn click(&mut self, p: vect2d) {
        self.move_to(p);
        self.button(p, true);
        self.button(p, false);
        self.time += 0.4;
    }

    /// Two clicks within egui's double-click window.
    pub fn double_click(&mut self, p: vect2d) {
        self.move_to(p);
        self.button(p, true);
        self.button(p, false);
        self.button(p, true);
        self.button(p, false);
        self.time += 0.4;
    }

    /// Press at `from`, move in steps, release at `to`. The step count
    /// keeps each frame's motion well past egui's drag threshold.
    pub fn drag(&mut self, from: vect2d, to: vect2d) {
        self.drag_moves(from, to);
        self.button(to, false);
    }

    /// Release the primary button at a sketch position (ends a
    /// gesture started with drag_moves).
    pub fn release(&mut self, p: vect2d) {
        self.button(p, false);
        self.time += 0.4;
    }

    /// The press-and-move half of a drag, no release -- for tests that
    /// interrupt the gesture (Escape, tool switch).
    pub fn drag_moves(&mut self, from: vect2d, to: vect2d) {
        self.move_to(from);
        self.button(from, true);
        for i in 1..=4 {
            let t = i as f64 / 4.0;
            let p = vect2d::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
            let sp = self.screen(p);
            self.events.push(egui::Event::PointerMoved(sp));
            self.frame();
        }
    }

    pub fn key(&mut self, key: egui::Key) {
        self.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: self.modifiers,
        });
        self.frame();
        self.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers: self.modifiers,
        });
        self.frame();
    }

    /// Key with modifiers held for the whole frame (Ctrl+Z etc.).
    pub fn key_with(&mut self, key: egui::Key, modifiers: egui::Modifiers) {
        let saved = self.modifiers;
        self.modifiers = modifiers;
        self.key(key);
        self.modifiers = saved;
    }

    /// Type text into whatever has keyboard focus.
    pub fn type_text(&mut self, s: &str) {
        self.events.push(egui::Event::Text(s.to_string()));
        self.frame();
    }

    // -- assertion helpers ------------------------------------------

    pub fn sketch(&self) -> &arael_sketch_solver::Sketch {
        &self.app.sketch
    }

    pub fn line_count(&self) -> usize {
        self.sketch().lines.len()
    }

    pub fn arc_count(&self) -> usize {
        self.sketch().arcs.len()
    }

    /// Endpoints of the i-th line in ref order.
    pub fn line(&self, i: usize) -> (vect2d, vect2d) {
        let r = self.sketch().lines.refs().nth(i).expect("line index");
        let l = &self.sketch().lines[r];
        (l.p1.value, l.p2.value)
    }

    /// State no frame may leave behind, whatever the test does.
    fn check_invariants(&self) {
        let s: &arael_sketch_solver::Sketch = &self.app.sketch;
        // Drag apparatus exists only during an active grab.
        if self.app.grab.is_none() {
            assert!(
                self.app.drag_apparatus.is_none(),
                "drag apparatus left behind with no active grab"
            );
        }
        // Selection refs must be valid.
        for sel in &self.app.selection {
            let ok = match *sel {
                Selection::Point(r) => s.points.contains_ref(r),
                Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) =>
                    s.lines.contains_ref(r),
                Selection::Arc(r) | Selection::ArcCenter(r)
                | Selection::ArcStart(r) | Selection::ArcEnd(r) =>
                    s.arcs.contains_ref(r),
                Selection::Dimension(did) => s.dimension_index_by_did(did).is_some(),
                Selection::Constraint(_) => true, // nid-addressed, prune handles staleness
            };
            assert!(ok, "stale selection entry {:?}", sel);
        }
        // History cursor stays in range.
        assert!(
            self.app.history.cursor <= self.app.history.actions.len(),
            "history cursor {} past {} actions",
            self.app.history.cursor,
            self.app.history.actions.len()
        );
        // Cost stays finite.
        assert!(self.app.last_cost.is_finite(), "non-finite cost");
    }
}

pub fn v(x: f64, y: f64) -> vect2d {
    vect2d::new(x, y)
}

/// Approximate point equality in sketch units.
pub fn near(a: vect2d, b: vect2d, tol: f64) -> bool {
    (a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol
}
