//! The per-call memo of a walk over an expression DAG: a node reached
//! along several paths is visited once and its result shared.

use super::{Expr, E};
use std::collections::HashMap;

/// A walk's results so far, keyed by node address. The value holds the
/// node itself, so the address stays unique while the memo lives.
pub(crate) type NodeMemo = HashMap<*const Expr, (E, E), std::hash::BuildHasherDefault<PtrHash>>;

/// Whether a node needs the memo: only one reached along several paths
/// does, and such a node holds more than one reference. A leaf's result
/// is as cheap to recompute as to look up. A fresh expression, whose
/// nodes are each held once, stays out of the map altogether.
pub(crate) fn shared(e: &E) -> bool {
    std::rc::Rc::strong_count(&e.0) > 1
        && !matches!(&*e.0, Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. })
}

/// A hasher for node addresses: the address with its alignment bits
/// dropped, spread by a multiply. The default hasher is built for
/// untrusted keys and costs more than the lookup it serves here.
#[derive(Default)]
pub(crate) struct PtrHash(u64);

impl std::hash::Hasher for PtrHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    fn write_usize(&mut self, p: usize) {
        self.0 = ((p as u64) >> 3).wrapping_mul(0x9E3779B97F4A7C15);
    }
}
