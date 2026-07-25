//! `cargo arael` -- generate C ABI + C++ (and later Python)
//! interfaces for arael root models. See docs/dev/CXX.md for the
//! design and docs/SIDECAR.md for the model description it consumes.

use cargo_arael::export;

const USAGE: &str = "\
usage: cargo arael <command> [options]

commands:
  export    build the model crate, harvest its sidecar, and (re)generate
            the interface tree (capi/, cxx/)
  check     regenerate in memory and fail if the committed tree is stale

options:
  --manifest-dir <path>   model crate directory (default: current)
  --root <name>           pick a root when the crate has several
";

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Invoked as `cargo arael ...`: cargo passes "arael" through.
    if args.first().map(String::as_str) == Some("arael") {
        args.remove(0);
    }
    let mut dir = std::path::PathBuf::from(".");
    let mut root: Option<String> = None;
    let mut cmd: Option<String> = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--manifest-dir" => {
                dir = it.next().map(Into::into).unwrap_or_else(|| {
                    eprintln!("--manifest-dir needs a value");
                    std::process::exit(2);
                });
            }
            "--root" => root = it.next(),
            "-h" | "--help" => {
                print!("{USAGE}");
                return;
            }
            c if cmd.is_none() && !c.starts_with('-') => cmd = Some(a),
            other => {
                eprintln!("unknown argument `{other}`\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    let result = match cmd.as_deref() {
        Some("export") => export::run_export(&dir, root.as_deref()),
        Some("check") => export::run_check(&dir, root.as_deref()),
        _ => {
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("cargo arael: {e}");
        std::process::exit(1);
    }
}
