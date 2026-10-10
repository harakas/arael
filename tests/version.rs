use arael::VERSION;

#[test]
fn version_matches_the_manifest() {
    let text = env!("CARGO_PKG_VERSION");
    assert_eq!(VERSION.to_string(), text);
    let (numbers, pre) = match text.split_once('-') {
        Some((n, p)) => (n, p),
        None => (text, ""),
    };
    let mut it = numbers.split('.').map(|s| s.parse::<u32>().unwrap());
    assert_eq!(VERSION.major, it.next().unwrap());
    assert_eq!(VERSION.minor, it.next().unwrap());
    assert_eq!(VERSION.patch, it.next().unwrap());
    assert_eq!(it.next(), None);
    assert_eq!(VERSION.pre, pre);
}
