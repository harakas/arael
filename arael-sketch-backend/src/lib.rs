//! Headless backend for arael-sketch.
//!
//! Contains the command parser, action pipeline, undo/redo history,
//! constraint-conflict detection, geometry helpers and MCP server --
//! everything that used to live in `arael-sketch` but does not touch
//! the GUI. Depends only on the solver (`arael-sketch-solver`) and
//! numerical deps; pulls in zero egui/eframe code.

pub mod geometry;
pub mod earc_fit;
pub mod ids;
pub mod actions;
pub mod history;
pub mod conflicts;
pub mod commands;
#[cfg(not(target_arch = "wasm32"))]
pub mod mcp_server;

pub use ids::{ConstraintId, CoincidentKind, MidpointKind, Selection, find_constraint_by_name};
pub use actions::{Action, resolve_dim_endpoint};
pub use history::{History, CursorState};
pub use conflicts::check_constraint_conflict;
pub use commands::{CommandContext, DRAG_PULL_WEIGHT};
