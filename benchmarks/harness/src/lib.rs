// The benchmark harness: what every arael benchmark does the same way.
//
// pgo, slam, loc and bal all measure the same thing -- what one iteration of a
// nonlinear solver costs on an identical problem -- and each of them had grown
// its own copy of the machinery. The copies drifted, and every timing bug found
// in pgo (a probe mutating the model, a first iteration full of rejected steps,
// retries counted for one system and not another, a metric that amortized setup
// into the per-iteration number) was a bug the others still have.
//
// A benchmark supplies its problem: how to build it, how to score a solution,
// which systems to run. Everything else lives here.

pub mod arael;
pub mod external;
pub mod header;
#[cfg(feature = "factrs")]
pub mod factrs;
pub mod mem;
pub mod pin;
pub mod probe;
pub mod solver;
pub mod table;
