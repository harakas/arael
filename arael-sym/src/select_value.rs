//! [`select`] over values made of scalars: vectors, matrices,
//! quaternions, tuples and arrays.

use crate::geo::{matrix2sym, matrix3sym, matrixsym, quaternsym, vect2sym, vect3sym, vectsym};
use crate::{select, E};

/// A value [`select_value`] switches: a scalar, or a value made of
/// scalars, switched one scalar component at a time.
pub trait Select: Sized {
    /// `arms[i]` for index `i`, else `default`, component by component.
    /// Every arm and the default have the same shape.
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self;
}

/// [`select`] for a value made of scalars: `arms[i]` for index `i`,
/// else `default`, every scalar component through its own `select` on
/// `index`. Under [`cse_scoped`](crate::cse_scoped) the selects on one
/// index emit as one `match`.
///
/// ```
/// use arael_sym::{select_value, symbol, vect2sym, E};
/// let a = vect2sym::new("a");
/// let b = vect2sym::new("b");
/// let (p, w): (vect2sym, E) = select_value(symbol("k"), vec![(a, 1.0.into()), (b, 2.0.into())], None);
/// assert_eq!(format!("{}", p.x), "select(k, a.x, b.x)");
/// assert_eq!(format!("{}", w), "select(k, 1, 2)");
/// ```
pub fn select_value<T: Select>(index: impl Into<E>, arms: Vec<T>, default: Option<T>) -> T {
    T::select_arms(&index.into(), arms, default)
}

/// What a `match` arm's Rust value switches as: numbers become [`E`],
/// the sym types stay, tuples and arrays convert element by element.
pub trait IntoSelect {
    /// The switched value.
    type Value: Select;
    /// Convert.
    fn into_select(self) -> Self::Value;
}

impl Select for E {
    fn select_arms(index: &E, arms: Vec<E>, default: Option<E>) -> E {
        select(index.clone(), arms, default)
    }
}

/// Split each arm (and the default) into its parts with `parts`, then
/// switch part `k` of every arm on its own: `out[k]` is that switch.
fn select_parts<T, P: Select>(
    index: &E, arms: Vec<T>, default: Option<T>, n: usize, parts: impl Fn(T) -> Vec<P>,
) -> Vec<P> {
    let mut by_part: Vec<Vec<P>> = (0..n).map(|_| Vec::with_capacity(arms.len())).collect();
    for arm in arms {
        for (k, p) in parts(arm).into_iter().enumerate() {
            by_part[k].push(p);
        }
    }
    let mut default_parts: Vec<Option<P>> = match default {
        Some(d) => parts(d).into_iter().map(Some).collect(),
        None => (0..n).map(|_| None).collect(),
    };
    by_part.into_iter().zip(default_parts.drain(..))
        .map(|(arms, d)| P::select_arms(index, arms, d))
        .collect()
}

impl Select for vect2sym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let mut c = select_parts(index, arms, default, 2, |v| vec![v.x, v.y]).into_iter();
        vect2sym::from_components(c.next().unwrap(), c.next().unwrap())
    }
}

impl Select for vect3sym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let mut c = select_parts(index, arms, default, 3, |v| vec![v.x, v.y, v.z]).into_iter();
        vect3sym::from_components(c.next().unwrap(), c.next().unwrap(), c.next().unwrap())
    }
}

impl Select for vectsym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let n = same_len(arms.iter().chain(default.iter()).map(|v| v.e.len()), "vectsym dimension");
        vectsym::from_components(select_parts(index, arms, default, n, |v| v.e))
    }
}

impl Select for matrix2sym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        matrix2sym { rows: <[vect2sym; 2]>::select_arms(index,
            arms.into_iter().map(|m| m.rows).collect(), default.map(|m| m.rows)) }
    }
}

impl Select for matrix3sym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        matrix3sym { rows: <[vect3sym; 3]>::select_arms(index,
            arms.into_iter().map(|m| m.rows).collect(), default.map(|m| m.rows)) }
    }
}

impl Select for matrixsym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let n = same_len(arms.iter().chain(default.iter()).map(|m| m.rows.len()), "matrixsym row count");
        matrixsym { rows: select_parts(index, arms, default, n, |m| m.rows) }
    }
}

impl Select for quaternsym {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let (ts, vs): (Vec<E>, Vec<vect3sym>) = arms.into_iter().map(|q| (q.t, q.v)).unzip();
        let (dt, dv) = match default {
            Some(q) => (Some(q.t), Some(q.v)),
            None => (None, None),
        };
        quaternsym { t: E::select_arms(index, ts, dt), v: vect3sym::select_arms(index, vs, dv) }
    }
}

impl<T: Select, const N: usize> Select for [T; N] {
    fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
        let parts = select_parts(index, arms, default, N, |a: [T; N]| Vec::from(a));
        match <[T; N]>::try_from(parts) {
            Ok(a) => a,
            Err(_) => unreachable!("select_parts yields N parts"),
        }
    }
}

/// The one length every item has; a panic naming the lengths otherwise.
fn same_len(mut lens: impl Iterator<Item = usize>, what: &str) -> usize {
    let first = lens.next().expect("select_value needs at least one arm or a default");
    for n in lens {
        assert_eq!(n, first, "select_value: arms of {} {} and {}", what, first, n);
    }
    first
}

macro_rules! select_tuple {
    ($($t:ident $a:ident $d:ident $i:tt),+) => {
        impl<$($t: Select),+> Select for ($($t,)+) {
            fn select_arms(index: &E, arms: Vec<Self>, default: Option<Self>) -> Self {
                $(let mut $a: Vec<$t> = Vec::with_capacity(arms.len());)+
                for arm in arms {
                    $($a.push(arm.$i);)+
                }
                $(let mut $d: Option<$t> = None;)+
                if let Some(d) = default {
                    $($d = Some(d.$i);)+
                }
                ($($t::select_arms(index, $a, $d),)+)
            }
        }
        impl<$($t: IntoSelect),+> IntoSelect for ($($t,)+) {
            type Value = ($($t::Value,)+);
            fn into_select(self) -> Self::Value {
                ($(self.$i.into_select(),)+)
            }
        }
    };
}

select_tuple!(A a0 d0 0);
select_tuple!(A a0 d0 0, B a1 d1 1);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6,
              I a7 d7 7);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6,
              I a7 d7 7, J a8 d8 8);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6,
              I a7 d7 7, J a8 d8 8, K a9 d9 9);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6,
              I a7 d7 7, J a8 d8 8, K a9 d9 9, L a10 d10 10);
select_tuple!(A a0 d0 0, B a1 d1 1, C a2 d2 2, D a3 d3 3, F a4 d4 4, G a5 d5 5, H a6 d6 6,
              I a7 d7 7, J a8 d8 8, K a9 d9 9, L a10 d10 10, M a11 d11 11);

macro_rules! into_select_number {
    ($($t:ty),+) => {$(
        impl IntoSelect for $t {
            type Value = E;
            fn into_select(self) -> E { E::from(self) }
        }
    )+};
}
into_select_number!(f64, i64, i32);

macro_rules! into_select_same {
    ($($t:ty),+) => {$(
        impl IntoSelect for $t {
            type Value = $t;
            fn into_select(self) -> $t { self }
        }
    )+};
}
into_select_same!(E, vect2sym, vect3sym, vectsym, matrix2sym, matrix3sym, matrixsym, quaternsym);

impl<T: IntoSelect, const N: usize> IntoSelect for [T; N] {
    type Value = [T::Value; N];
    fn into_select(self) -> Self::Value { self.map(IntoSelect::into_select) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol;

    fn show(e: &E) -> String { format!("{}", e) }

    #[test]
    fn a_vector_switches_component_by_component() {
        let k = symbol("k");
        let v = select_value(k.clone(), vec![vect3sym::new("a"), vect3sym::new("b")], Some(vect3sym::new("c")));
        assert_eq!(show(&v.x), show(&select(k.clone(), vec![symbol("a.x"), symbol("b.x")], Some(symbol("c.x")))));
        assert_eq!(show(&v.z), show(&select(k, vec![symbol("a.z"), symbol("b.z")], Some(symbol("c.z")))));
    }

    #[test]
    fn a_tuple_and_an_array_switch_element_by_element() {
        let k = symbol("k");
        let (p, w) = select_value(k.clone(),
            vec![(vect2sym::new("a"), symbol("u")), (vect2sym::new("b"), symbol("v"))], None);
        assert_eq!(show(&p.y), "select(k, a.y, b.y)");
        assert_eq!(show(&w), "select(k, u, v)");
        let [r0, r1] = select_value(k, vec![[symbol("a"), symbol("b")], [symbol("c"), symbol("d")]], None);
        assert_eq!(show(&r0), "select(k, a, c)");
        assert_eq!(show(&r1), "select(k, b, d)");
    }

    #[test]
    fn matrices_and_quaternions_switch_every_entry() {
        let k = symbol("k");
        let m = select_value(k.clone(), vec![matrix2sym::new("a"), matrix2sym::new("b")], None);
        assert_eq!(show(&m.rows[1].x), show(&select(k.clone(), vec![
            matrix2sym::new("a").rows[1].x.clone(), matrix2sym::new("b").rows[1].x.clone()], None)));
        let q = select_value(k.clone(), vec![quaternsym::new("p"), quaternsym::new("q")], None);
        assert_eq!(show(&q.v.y), show(&select(k.clone(), vec![
            quaternsym::new("p").v.y, quaternsym::new("q").v.y], None)));
        let n = select_value(k, vec![matrixsym::new("a", 2, 4), matrixsym::new("b", 2, 4)], None);
        assert_eq!((n.rows.len(), n.rows[1].e.len()), (2, 4));
    }

    #[test]
    fn numbers_convert_to_scalars() {
        let arms = vec![(1.0, vect2sym::new("a")).into_select(), (2i32, vect2sym::new("b")).into_select()];
        let (w, _) = select_value(symbol("k"), arms, Some((3i64, vect2sym::new("c")).into_select()));
        assert_eq!(show(&w), "select_or(k, 1, 2, 3)");
    }

    #[test]
    #[should_panic(expected = "vectsym dimension 3 and 4")]
    fn vectors_of_different_dimensions_panic() {
        select_value(symbol("k"), vec![vectsym::new("a", 3), vectsym::new("b", 4)], None);
    }
}
