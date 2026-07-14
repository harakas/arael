//! Where arael's diagnostics go. Messages print to stderr unless a level or a
//! sink says otherwise.
//!
//! - [`set_level`] / [`silence`] -- drop everything above a level. Checked at
//!   the call site before the message is formatted, so a silenced arael
//!   allocates nothing.
//! - [`set_sink`] / [`reset_sink`] -- send messages somewhere other than stderr.
//!
//! ```no_run
//! use arael::log::{self, Level};
//!
//! log::silence();                     // emit nothing
//! log::set_level(Level::Error);       // errors only
//! log::set_sink(|level, msg| eprintln!("{}: {}", level.tag(), msg));
//! log::reset_sink();                  // back to stderr
//! ```

use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Severity of a message, and the threshold [`set_level`] compares against.
/// Ordered `Off < Error < Warn < Info`: a level admits everything at or below
/// itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// Nothing is emitted.
    Off = 0,
    /// The solve is giving up or returning early.
    Error = 1,
    /// The solve continues, but something is suspect: a saturated `safe_sqrt`,
    /// a non-unit quaternion.
    Warn = 2,
    /// The `verbose` trace, and the backend reporting what it chose.
    Info = 3,
}

impl Level {
    /// The tag the built-in stderr sink prints.
    pub fn tag(self) -> &'static str {
        match self {
            Level::Off => "OFF",
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
        }
    }
}

/// Default: everything.
static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

type Sink = Box<dyn Fn(Level, &str) + Send + Sync>;

/// `None` means the built-in stderr sink.
static SINK: RwLock<Option<Sink>> = RwLock::new(None);

/// Drop every message above `level`. `Level::Off` silences arael entirely.
pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// The level currently in force.
pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        0 => Level::Off,
        1 => Level::Error,
        2 => Level::Warn,
        _ => Level::Info,
    }
}

/// Emit nothing at all. Same as `set_level(Level::Off)`.
pub fn silence() {
    set_level(Level::Off);
}

/// Would a message at this level be emitted? Public because the logging macros
/// expand to it, calling it before they format their arguments.
#[inline]
pub fn enabled(level: Level) -> bool {
    (level as u8) <= LEVEL.load(Ordering::Relaxed)
}

/// Send messages here instead of to stderr. The sink is called with the message
/// already formatted, on whichever thread logged it.
pub fn set_sink(sink: impl Fn(Level, &str) + Send + Sync + 'static) {
    if let Ok(mut slot) = SINK.write() {
        *slot = Some(Box::new(sink));
    }
}

/// Put the built-in stderr sink back.
pub fn reset_sink() {
    if let Ok(mut slot) = SINK.write() {
        *slot = None;
    }
}

/// Emit one message. The logging macros expand to this.
pub fn emit(level: Level, msg: &str) {
    if !enabled(level) {
        return;
    }
    // A poisoned lock falls through to stderr rather than panicking mid-solve.
    if let Ok(sink) = SINK.read()
        && let Some(f) = sink.as_ref()
    {
        f(level, msg);
        return;
    }
    eprintln!("[arael {}] {}", level.tag(), msg);
}

/// Log an informational message.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Info) {
            $crate::log::emit($crate::log::Level::Info, &format!($($arg)*));
        }
    };
}

/// Log a warning.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Warn) {
            $crate::log::emit($crate::log::Level::Warn, &format!($($arg)*));
        }
    };
}

/// Log an error.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Error) {
            $crate::log::emit($crate::log::Level::Error, &format!($($arg)*));
        }
    };
}
