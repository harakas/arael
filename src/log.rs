/// Log an informational message to stderr.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        eprintln!("[arael INFO] {}", format!($($arg)*))
    };
}

/// Log a warning message to stderr.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        eprintln!("[arael WARN] {}", format!($($arg)*))
    };
}

/// Log an error message to stderr.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        eprintln!("[arael ERROR] {}", format!($($arg)*))
    };
}
