 * Do not use unicode in comments or debug prints.
 * Commit only when I request it.
 * When finding a bug/issue and adding a fix, also add a test for it.
 * When adding a new feature -- add tests for it.
 * We use following euler angles and coordinate system convention: x is roll, y is pitch, z is yaw. Axes are x=forward, y=left, z=up. And rotation=rot(ea.z)*rot(ea.y)*rot(ea.x).
 * When committing a sub-crate/component, prefix commit message with `<component name>: message`
 * When updating README.md also update src/lib.rs crate documentation -- and the other way around
 * Run `cargo audit` periodically to check dependencies for known vulnerabilities.
 * Arael-sketch has command interface and MCP server to allow outside agents to use it. It is documented in ./arael-sketch-backend/docs/COMMANDS.md -- when adding new ui features, also add commands to accomplish the same programmatically/by agent.
 * When leaving things unimplemented in the plan, add them into TODO.md with explanation why they weren't implemented.
 * When adding/modifying arael-sketch commands, also update the documentation in COMMANDS.md.
 * Arael-sketch architecture: commands -> actions -> sketch (solver). Both the GUI and the command interface call actions. Actions mutate the sketch. Never call commands from actions or the GUI -- always go through actions. Helper points (Pc) used internally by constraints are created inside action apply() methods, invisible to callers.
 * arael-sketch: you can run test scripts inline with stdout and nogui: `cargo run -r -p arael-sketch -- --empty --nogui --stdout --script script.cmd`
 * Never hide errors. One bug does not justify another or klunky workarounds -- fix issues at the source or prompt for decision by user.
 * When something is "impossible" to fix due to limited API, extend the API/refactor the code instead of working around or ignoring problems and creating more bugs and bad code. A small change might cause a big change -- that is fine if the architectural changes needed take us forward.
 * arael-sketch: having to use the `force` keyword on a constraint to take hold (rejected otherwise for not changing dof) means that either we have a bug or our assumptions about the constraint are invalid and it actually does not constrain anything. So don't go adding force to places and handwaving it away with "degenerate geometry so force needed".
 * When a test fails then it's probably a bug in changes. First instinct should not be to "fix" the test but to understand why it fails. Hallucinating plausible but unverified reasons from local changes is not acceptable. Causes must be verified.
 * arael-sketch: We have color scheme in colors.rs -- so don't hardcode colors but add/reuse entries.
 * There is no need to add #[arael(skip)] on every non-parameter bool, f32/f64, String, etc. They just clutter the code and reduce readability.
 * When modifying arael-sym with new features and docs -- also update docs/SYM.md
 * When adding/changing features that might have performance impacts, run before and after benchmarks to compare, reject regressions in performance, and flag improvements.
 * When benchmark numbers change, regenerate the charts (`benchmarks/make_slam_loc_chart.py`, `benchmarks/pgo/make_chart.py`)
 * Charts contain release version, so need to be updated when they have changed for a new release
 * Charts live in `benchmarks/charts/v<version>/` -- the path carries the version. Do not delete legacy versions as they are referenced by published crates.
 * README.md and src/lib.rs point at the current version's chart dir -- update on release
