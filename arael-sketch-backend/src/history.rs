//! Undo/redo stack and canonical edit log for a sketch.
//!
//! `History` exists for four reasons:
//!
//! 1. **Undo/redo.** Every GUI toolbar click, drag, and
//!    command-panel line funnels through [`History::push`]. Ctrl+Z
//!    calls [`History::undo`], Ctrl+Shift+Z calls
//!    [`History::redo`]. Without `History`, the editor has no
//!    reversible edits.
//!
//! 2. **Atomicity across multi-step operations.**
//!    [`History::begin_group`] tags subsequent pushes with the same
//!    group id. A rectangle tool that pushes four `AddLine`s, four
//!    `ApplyCoincidentLL21`s, and two length flags is *one* undo
//!    frame, not ten. `Action::Drag { snapshot }` likewise rolls an
//!    entire drag trajectory into one reversible unit.
//!
//! 3. **Cursor restoration.** Each frame stores a [`CursorState`]
//!    -- where the command-panel cursor was and which tangent it
//!    pointed along. Undoing a `move` command puts the cursor back
//!    where it was so the user can keep typing naturally, instead
//!    of dropping it to the origin.
//!
//! 4. **Snapshot storage, not replay.** Each push serialises the
//!    post-apply sketch via bincode. Undo/redo deserialises the
//!    snapshot and calls `sketch.solve()` once; it does *not*
//!    replay the action log forward. This is deliberate:
//!    `Action::apply` is non-deterministic in general (drag
//!    trajectories, solver starting points, helper-point ordering),
//!    so a forward replay could diverge. Snapshots make undo/redo
//!    exact.
//!
//! The cost is that every push serialises the whole sketch. In
//! practice that is tens to low hundreds of kilobytes of bincode
//! and has been fine for interactive use. If that ever becomes a
//! bottleneck -- for instance, recording thousands of frames of
//! procedural generation -- the likely fix is a delta-snapshot
//! scheme or a determinism contract on `Action::apply` that lets
//! replay take over.
//!
//! # Only `Action`s are tracked
//!
//! **Any mutation you want to be reversible must go through an
//! [`Action`].** Direct mutation of sketch fields --
//! `sketch.lines[r].p1 = Param::fixed(..)`, pushing raw constraint
//! structs into the collection vectors, flipping
//! `sketch.cached_dof`, etc. -- bypasses `History` entirely, is
//! invisible to undo/redo, and will not come back on a redo.
//!
//! The raw `Sketch` API is there for headless batch work where
//! history does not matter (see the `rectangle_solver` example).
//! The moment you want undo/redo, always go through the `Action`
//! enum (see the `rectangle_actions` example, including the
//! `LockLineP1` action used instead of a direct `Param::fixed`
//! assignment).

use arael::vect::vect2d;
use arael_sketch_solver::Sketch;
use crate::actions::Action;

/// Cursor state saved alongside each history snapshot.
#[derive(Clone, Default)]
pub struct CursorState {
    pub pos: Option<vect2d>,
    pub tangent: Option<vect2d>,
}

pub struct History {
    pub actions: Vec<Action>,
    pub snapshots: Vec<Vec<u8>>,  // bincode-serialized Sketch after each action
    pub cursors: Vec<CursorState>,
    pub groups: Vec<u32>,         // group id for each action
    pub cursor: usize,            // number of applied actions (0 = initial state)
    pub next_group: u32,
    pub current_group: u32,
    initial_snapshot: Vec<u8>,    // state before any actions
}

impl History {
    pub fn new(sketch: &Sketch) -> Self {
        History {
            actions: Vec::new(), snapshots: Vec::new(), cursors: Vec::new(), groups: Vec::new(),
            // Pushes before the first begin_group carry group 0; the
            // first explicit group must not share that id.
            cursor: 0, next_group: 1, current_group: 0,
            initial_snapshot: bincode::serialize(sketch).unwrap(),
        }
    }

    pub fn begin_group(&mut self) {
        self.current_group = self.next_group;
        self.next_group += 1;
    }

    pub fn push(&mut self, action: Action, sketch: &Sketch, cursor: CursorState) {
        // Truncate any redo tail
        self.actions.truncate(self.cursor);
        self.snapshots.truncate(self.cursor);
        self.cursors.truncate(self.cursor);
        self.groups.truncate(self.cursor);
        // Push new
        self.actions.push(action);
        self.snapshots.push(bincode::serialize(sketch).unwrap());
        self.cursors.push(cursor);
        self.groups.push(self.current_group);
        self.cursor += 1;
    }

    pub fn can_undo(&self) -> bool { self.cursor > 0 }
    pub fn can_redo(&self) -> bool { self.cursor < self.actions.len() }

    pub fn undo(&mut self) -> Option<(Sketch, CursorState)> {
        if self.cursor == 0 { return None; }
        // The cursor state to restore is saved at the start of the group being undone
        let group = self.groups[self.cursor - 1];
        // Find the first action of this group to get the pre-group cursor
        let mut group_start = self.cursor - 1;
        while group_start > 0 && self.groups[group_start - 1] == group {
            group_start -= 1;
        }
        let restored_cursor = self.cursors[group_start].clone();
        // Rewind cursor to before the group
        while self.cursor > 0 && self.groups[self.cursor - 1] == group {
            self.cursor -= 1;
        }
        if self.cursor == 0 {
            let mut sketch: Sketch = bincode::deserialize(&self.initial_snapshot).unwrap();
            sketch.solve();
            Some((sketch, restored_cursor))
        } else {
            let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
            sketch.solve();
            Some((sketch, restored_cursor))
        }
    }

    pub fn redo(&mut self) -> Option<(Sketch, CursorState)> {
        if self.cursor >= self.actions.len() { return None; }
        // Find the end of the next group
        let group = self.groups[self.cursor];
        while self.cursor < self.actions.len() && self.groups[self.cursor] == group {
            self.cursor += 1;
        }
        let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
        sketch.solve();
        // Each saved cursor state is the one from just before its
        // action, so the state after the redone group is the next
        // entry's. For the newest group no later entry exists; the
        // pre-last-action state is the closest on record.
        let restored_cursor = self.cursors.get(self.cursor).cloned()
            .unwrap_or_else(|| self.cursors[self.cursor - 1].clone());
        Some((sketch, restored_cursor))
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
    /// Returns the resulting sketch + cursor, or None if position is invalid.
    pub fn goto(&mut self, target: usize) -> Option<(Sketch, CursorState)> {
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
