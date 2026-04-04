 * Do not use unicode in comments or debug prints.
 * Commit only when I request it.
 * When finding a bug/issue and adding a fix, also add a test for it.
 * When adding a new feature -- add tests for it.
 * We use following euler angles and coordinate system convention: x is roll, y is pitch, z is yaw. Axes are x=forward, y=left, z=up. And rotation=rot(ea.z)*rot(ea.y)*rot(ea.x).
 * When committing a sub-crate/component, prefix commit message with `<component name>: message`
 * When updating README.md also update src/lib.rs crate documentation -- and the other way around
 * Run `cargo audit` periodically to check dependencies for known vulnerabilities.
 * Arael-sketch has command interface and MCP server to allow outside agents to use it. It is documented in ./arael-sketch/docs/COMMANDS.md -- when adding new ui features, also add commands to accomplish the same programmatically/by agent.
 * For AI agents the MCP server sends a small command overview at init -- keep it up to date as we add/change features.
 * When leaving things unimplemented in the plan, add them into TODO.md with explanation why they weren't implemented.
 * When adding/modifying arael-sketch commands, also update the documentation in COMMANDS.md.
 * Arael-sketch architecture: commands -> actions -> sketch (solver). Both the GUI and the command interface call actions. Actions mutate the sketch. Never call commands from actions or the GUI -- always go through actions. Helper points (Pc) used internally by constraints are created inside action apply() methods, invisible to callers.
 * arael-sketch: you can run test scripts inline with stdout and nogui: `cargo run -r -p arael-sketch -- --empty --nogui --stdout --script script.cmd`
