// History for undo/redo in the sketch editor.

use arael_sketch_solver::Sketch;
use crate::actions::Action;

pub struct History {
    pub actions: Vec<Action>,
    pub snapshots: Vec<Vec<u8>>,  // bincode-serialized Sketch after each action
    pub groups: Vec<u32>,         // group id for each action
    pub cursor: usize,            // number of applied actions (0 = empty sketch)
    pub next_group: u32,
    pub current_group: u32,
}

impl History {
    pub fn new() -> Self {
        History {
            actions: Vec::new(), snapshots: Vec::new(), groups: Vec::new(),
            cursor: 0, next_group: 0, current_group: 0,
        }
    }

    pub fn begin_group(&mut self) {
        self.current_group = self.next_group;
        self.next_group += 1;
    }

    pub fn push(&mut self, action: Action, sketch: &Sketch) {
        // Truncate any redo tail
        self.actions.truncate(self.cursor);
        self.snapshots.truncate(self.cursor);
        self.groups.truncate(self.cursor);
        // Push new
        self.actions.push(action);
        self.snapshots.push(bincode::serialize(sketch).unwrap());
        self.groups.push(self.current_group);
        self.cursor += 1;
    }

    pub fn can_undo(&self) -> bool { self.cursor > 0 }
    pub fn can_redo(&self) -> bool { self.cursor < self.actions.len() }

    pub fn undo(&mut self) -> Option<Sketch> {
        if self.cursor == 0 { return None; }
        // Find the start of the current group
        let group = self.groups[self.cursor - 1];
        while self.cursor > 0 && self.groups[self.cursor - 1] == group {
            self.cursor -= 1;
        }
        if self.cursor == 0 {
            Some(Sketch::new())
        } else {
            let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
            sketch.solve();
            Some(sketch)
        }
    }

    pub fn redo(&mut self) -> Option<Sketch> {
        if self.cursor >= self.actions.len() { return None; }
        // Find the end of the next group
        let group = self.groups[self.cursor];
        while self.cursor < self.actions.len() && self.groups[self.cursor] == group {
            self.cursor += 1;
        }
        let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
        sketch.solve();
        Some(sketch)
    }
}
