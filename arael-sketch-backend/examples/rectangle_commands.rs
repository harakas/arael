//! Rectangle via the text-command API.
//!
//! Same rectangle as `rectangle_actions`, but driven by the
//! backend's `CommandContext` and parser -- every line is exactly
//! what the user would type into the `/` command panel in the GUI
//! or send via an MCP tool call. No hand-picked `Action` variants,
//! no `Ref`s threaded through our code; the parser resolves names
//! (`L0`, `L0.p2`, `midpoint(L0)`, ...) against the live sketch.
//!
//! Run with:
//!   cargo run -r -p arael-sketch-backend --example rectangle_commands
//!
//! The commands below are precisely what you would script in a
//! `.cmd` file or send from an MCP agent; the only thing the
//! example does on top of that is print each command and its
//! output so the transcript is easy to follow.

use arael_sketch_backend::CommandContext;
use arael_sketch_backend::commands;

fn main() {
    let mut ctx = CommandContext::new();

    let script = [
        // Four lines around the corners; deliberately off so the
        // solver has work to do.
        "add_line 0,0 3,0.1",
        "add_line L0.p2 3,2.1",
        "add_line L1.p2 0.1,1.9",
        "add_line L2.p2 L0.p1",
        // Horizontal / vertical flags.
        "horizontal L0",
        "horizontal L2",
        "vertical L1",
        "vertical L3",
        // Corner coincidences are auto-detected by add_line when an
        // endpoint snaps to an existing point, so the four corners
        // are already stitched. You can still add them explicitly
        // (the parser deduplicates with a hint).
        //
        // Fix one corner so the sketch cannot translate, then set
        // the two length dimensions.
        "lock L0.p1 0,0",
        "length L0 4",
        "length L3 2",
        // Inspect the result.
        "dof",
        "list",
    ];

    for line in script {
        println!("> {line}");
        for r in commands::execute(&mut ctx, line) {
            if !r.output.is_empty() && !r.no_echo {
                println!("{}", r.output);
            }
        }
    }
}
