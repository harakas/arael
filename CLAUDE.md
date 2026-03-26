 * Do not use unicode in comments or debug prints.
 * Commit only when I request it.
 * When finding a bug/issue and adding a fix, also add a test for it.
 * When adding a new feature -- add tests for it.
 * We use following euler angles and coordinate system convention: x is roll, y is pitch, z is yaw. Axes are x=forward, y=left, z=up.
 * When committing a sub-crate/component, prefix commit message with `<component name>: message`
 * When updating README.md also update src/lib.rs crate documentation -- and the other way around
 * Run `cargo audit` periodically to check dependencies for known vulnerabilities.
