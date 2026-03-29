// History for undo/redo in the sketch editor.

use arael_sketch_solver::Sketch;
use crate::actions::Action;

pub struct History {
    pub actions: Vec<Action>,
    pub snapshots: Vec<Vec<u8>>,  // bincode-serialized Sketch after each action
    pub groups: Vec<u32>,         // group id for each action
    pub cursor: usize,            // number of applied actions (0 = initial state)
    pub next_group: u32,
    pub current_group: u32,
    initial_snapshot: Vec<u8>,    // state before any actions
}

impl History {
    pub fn new(sketch: &Sketch) -> Self {
        History {
            actions: Vec::new(), snapshots: Vec::new(), groups: Vec::new(),
            cursor: 0, next_group: 0, current_group: 0,
            initial_snapshot: bincode::serialize(sketch).unwrap(),
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
            let mut sketch: Sketch = bincode::deserialize(&self.initial_snapshot).unwrap();
            sketch.solve();
            Some(sketch)
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

    /// Return a list of (group_id, end_position, first_action_description) for
    /// all groups in the history. Position is the cursor value after the group.
    pub fn group_list(&self) -> Vec<(u32, usize, String)> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < self.actions.len() {
            let gid = self.groups[i];
            let desc = self.actions[i].describe();
            let mut count = 0;
            while i < self.actions.len() && self.groups[i] == gid {
                i += 1;
                count += 1;
            }
            let label = if count > 1 { format!("{} (+{})", desc, count - 1) } else { desc };
            result.push((gid, i, label));
        }
        result
    }

    /// Undo/redo to reach the given cursor position.
    /// Returns the resulting sketch, or None if position is invalid.
    pub fn goto(&mut self, target: usize) -> Option<Sketch> {
        let mut result = None;
        while self.cursor > target {
            result = self.undo();
            if result.is_none() { break; }
        }
        while self.cursor < target {
            result = self.redo();
            if result.is_none() { break; }
        }
        result
    }
}
