// `cargo arael export` gives the macro crates an optimized build in
// both profiles. Cargo builds proc-macros unoptimized in every profile
// and a model's code generation runs inside them, so a large model
// would otherwise spend most of its build expanding. Each of the four
// (profile, crate) entries is checked on its own: a manifest that has
// some of them, in the per-package or the build-override form, gets
// only the rest.

use cargo_arael::export::{ensure_macro_profiles, has_macro_profiles, missing_macro_profiles, run_setup};

const PLAIN: &str = "[package]\nname = \"m\"\nversion = \"0.1.0\"\n";

#[test]
fn a_plain_manifest_lacks_all_four() {
    assert_eq!(missing_macro_profiles(PLAIN), vec![
        ("dev", "arael-macros"), ("dev", "arael-sym"),
        ("release", "arael-macros"), ("release", "arael-sym"),
    ]);
}

#[test]
fn the_dev_entries_alone_leave_release_missing() {
    let t = format!("{PLAIN}\n[profile.dev.package.arael-macros]\nopt-level = 3\n\
                     [profile.dev.package.arael-sym]\nopt-level = 3\n");
    assert_eq!(missing_macro_profiles(&t),
               vec![("release", "arael-macros"), ("release", "arael-sym")]);
}

#[test]
fn a_build_override_covers_both_crates_of_its_profile() {
    let t = format!("{PLAIN}\n[profile.release.build-override]\nopt-level = 3\n");
    assert_eq!(missing_macro_profiles(&t),
               vec![("dev", "arael-macros"), ("dev", "arael-sym")]);
}

#[test]
fn all_four_per_package_entries_satisfy() {
    let t = format!("{PLAIN}\n\
        [profile.dev.package.arael-macros]\nopt-level = 3\n\
        [profile.dev.package.arael-sym]\nopt-level = 3\n\
        [profile.release.package.arael-macros]\nopt-level = 3\n\
        [profile.release.package.arael-sym]\nopt-level = 3\n");
    assert!(has_macro_profiles(&t));
}

#[test]
fn quoted_keys_count() {
    let t = format!("{PLAIN}\n\
        [profile.dev.package.\"arael-macros\"]\nopt-level = 3\n\
        [profile.dev.package.\"arael-sym\"]\nopt-level = 3\n\
        [profile.\"release\".build-override]\nopt-level = 3\n");
    assert!(has_macro_profiles(&t));
}

#[test]
fn an_unoptimized_setting_does_not_count() {
    let t = format!("{PLAIN}\n[profile.dev.package.arael-macros]\nopt-level = 0\n");
    assert!(missing_macro_profiles(&t).contains(&("dev", "arael-macros")));
}

#[test]
fn another_packages_opt_level_does_not_count() {
    let t = format!("{PLAIN}\n[profile.dev.package.serde]\nopt-level = 3\n");
    assert_eq!(missing_macro_profiles(&t).len(), 4);
}

#[test]
fn export_adds_only_the_missing_entries_and_is_idempotent() {
    let dir = std::env::temp_dir().join(format!("arael-mp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    // The README's old advice: dev only.
    let start = format!("{PLAIN}\n[profile.dev.package.arael-macros]\nopt-level = 3\n\
                         [profile.dev.package.arael-sym]\nopt-level = 3\n");
    std::fs::write(dir.join("Cargo.toml"), &start).unwrap();
    std::fs::write(dir.join("src/lib.rs"), "").unwrap();

    ensure_macro_profiles(&dir);
    let after = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
    assert!(has_macro_profiles(&after), "release entries were not added:\n{after}");
    assert!(after.starts_with(&start), "the existing manifest was not preserved");
    assert_eq!(after.matches("[profile.dev.package.arael-macros]").count(), 1,
               "a dev entry was added again");
    assert_eq!(after.matches("[profile.release.package.arael-sym]").count(), 1);

    ensure_macro_profiles(&dir);
    let twice = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
    assert_eq!(after, twice, "a second export changed the manifest");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn setup_adds_the_entries_and_fails_loudly_without_a_manifest() {
    let dir = std::env::temp_dir().join(format!("arael-setup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), PLAIN).unwrap();
    std::fs::write(dir.join("src/lib.rs"), "").unwrap();

    run_setup(&dir).unwrap();
    let after = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
    assert!(has_macro_profiles(&after));
    // Asked for on its own, a manifest that needs nothing is still success.
    run_setup(&dir).unwrap();

    std::fs::remove_dir_all(&dir).unwrap();
    // Unlike the export's silent path, setup reports a missing manifest.
    assert!(run_setup(&dir).is_err());
}
