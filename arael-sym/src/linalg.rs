use std::collections::HashMap;
use std::fmt;
use std::ops;
use super::{E, constant};
use super::diff::DiffVar;

// ============================================================
// SymVec
// ============================================================

/// Symbolic vector of expressions.
///
/// Supports element-wise arithmetic, dot product, differentiation,
/// evaluation, and code generation. Indexing is zero-based.
#[derive(Debug, Clone, PartialEq)]
pub struct SymVec(pub Vec<E>);

impl SymVec {
    /// Create a symbolic vector from a list of expressions.
    pub fn new(elems: Vec<E>) -> Self {
        SymVec(elems)
    }

    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to the element at index `i`.
    pub fn get(&self, i: usize) -> &E {
        &self.0[i]
    }

    /// Compute the dot product with another symbolic vector.
    pub fn dot(&self, other: &SymVec) -> E {
        assert_eq!(self.len(), other.len(), "dot product: length mismatch");
        let mut terms: Vec<E> = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            terms.push(self.0[i].clone() * other.0[i].clone());
        }
        terms.into_iter().reduce(|a, b| a + b).unwrap_or_else(|| constant(0.0))
    }

    /// Differentiate each element with respect to a variable.
    pub fn diff(&self, var: impl DiffVar) -> SymVec {
        let v = var.var_name();
        SymVec(self.0.iter().map(|e| e.diff(v)).collect())
    }

    /// Evaluate each element numerically given variable bindings.
    pub fn eval(&self, vars: &HashMap<&str, f64>) -> Vec<f64> {
        self.0.iter().map(|e| e.eval(vars)).collect()
    }

    /// Simplify each element.
    pub fn simplify(&self) -> SymVec {
        SymVec(self.0.iter().map(|e| e.simplify()).collect())
    }

    /// Expand each element (distribute products over sums).
    pub fn expand(&self) -> SymVec {
        SymVec(self.0.iter().map(|e| e.expand()).collect())
    }

    /// Substitute a variable in each element.
    pub fn subs(&self, var: &str, replacement: &E) -> SymVec {
        SymVec(self.0.iter().map(|e| e.subs(var, replacement)).collect())
    }

    /// Format the vector as a LaTeX column vector (pmatrix).
    pub fn to_latex(&self) -> String {
        let mut buf = String::from("\\begin{pmatrix} ");
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 { buf.push_str(" \\\\ "); }
            buf.push_str(&e.to_latex());
        }
        buf.push_str(" \\end{pmatrix}");
        buf
    }

    /// Generate Rust source code for the vector as a literal array.
    pub fn to_rust(&self, ft: &str) -> String {
        let mut buf = String::from("[");
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 { buf.push_str(", "); }
            buf.push_str(&e.to_rust(ft));
        }
        buf.push(']');
        buf
    }
}

impl ops::Index<usize> for SymVec {
    type Output = E;
    fn index(&self, i: usize) -> &E {
        &self.0[i]
    }
}

impl ops::Add for SymVec {
    type Output = SymVec;
    fn add(self, rhs: SymVec) -> SymVec {
        assert_eq!(self.len(), rhs.len(), "SymVec add: length mismatch");
        SymVec(
            self.0.into_iter().zip(rhs.0.into_iter())
                .map(|(a, b)| a + b)
                .collect()
        )
    }
}

impl ops::Mul<E> for SymVec {
    type Output = SymVec;
    fn mul(self, rhs: E) -> SymVec {
        SymVec(self.0.into_iter().map(|e| e * rhs.clone()).collect())
    }
}

impl ops::Mul<SymVec> for E {
    type Output = SymVec;
    fn mul(self, rhs: SymVec) -> SymVec {
        SymVec(rhs.0.into_iter().map(|e| self.clone() * e).collect())
    }
}

impl fmt::Display for SymVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            fmt::Display::fmt(e, f)?;
        }
        write!(f, "]")
    }
}

// ============================================================
// SymMat
// ============================================================

/// Symbolic matrix of expressions.
///
/// Stored in row-major order. Supports transpose, matrix multiplication,
/// matrix-vector product, element-wise differentiation, and code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SymMat {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major element data.
    pub data: Vec<E>,
}

impl SymMat {
    /// Create a matrix from dimensions and row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<E>) -> Self {
        assert_eq!(data.len(), rows * cols, "SymMat::new: data size mismatch");
        SymMat { rows, cols, data }
    }

    /// Create a zero matrix of the given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        SymMat {
            rows,
            cols,
            data: vec![constant(0.0); rows * cols],
        }
    }

    /// Create an n-by-n identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![constant(0.0); n * n];
        for i in 0..n {
            data[i * n + i] = constant(1.0);
        }
        SymMat { rows: n, cols: n, data }
    }

    /// Get a reference to the element at row `i`, column `j`.
    pub fn get(&self, i: usize, j: usize) -> &E {
        &self.data[i * self.cols + j]
    }

    /// Set the element at row `i`, column `j`.
    pub fn set(&mut self, i: usize, j: usize, val: E) {
        self.data[i * self.cols + j] = val;
    }

    /// Return the transpose of this matrix.
    pub fn transpose(&self) -> SymMat {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.get(i, j).clone());
            }
        }
        SymMat { rows: self.cols, cols: self.rows, data }
    }

    /// Differentiate every element with respect to a variable.
    pub fn diff(&self, var: impl DiffVar) -> SymMat {
        let v = var.var_name();
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|e| e.diff(v)).collect(),
        }
    }

    /// Evaluate every element numerically, returning a nested `Vec<Vec<f64>>`.
    pub fn eval(&self, vars: &HashMap<&str, f64>) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self.get(i, j).eval(vars));
            }
            result.push(row);
        }
        result
    }

    /// Simplify every element.
    pub fn simplify(&self) -> SymMat {
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|e| e.simplify()).collect(),
        }
    }

    /// Expand every element (distribute products over sums).
    pub fn expand(&self) -> SymMat {
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|e| e.expand()).collect(),
        }
    }

    /// Substitute a variable in every element.
    pub fn subs(&self, var: &str, replacement: &E) -> SymMat {
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|e| e.subs(var, replacement)).collect(),
        }
    }

    /// Format the matrix as a LaTeX pmatrix.
    pub fn to_latex(&self) -> String {
        let mut buf = String::from("\\begin{pmatrix} ");
        for i in 0..self.rows {
            if i > 0 { buf.push_str(" \\\\ "); }
            for j in 0..self.cols {
                if j > 0 { buf.push_str(" & "); }
                buf.push_str(&self.get(i, j).to_latex());
            }
        }
        buf.push_str(" \\end{pmatrix}");
        buf
    }

    /// Generate Rust source code for the matrix as a nested array literal.
    pub fn to_rust(&self, ft: &str) -> String {
        let mut buf = String::from("[");
        for i in 0..self.rows {
            if i > 0 { buf.push_str(", "); }
            buf.push('[');
            for j in 0..self.cols {
                if j > 0 { buf.push_str(", "); }
                buf.push_str(&self.get(i, j).to_rust(ft));
            }
            buf.push(']');
        }
        buf.push(']');
        buf
    }
}

// SymMat + SymMat
impl ops::Add for SymMat {
    type Output = SymMat;
    fn add(self, rhs: SymMat) -> SymMat {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols), "SymMat add: dimension mismatch");
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.into_iter().zip(rhs.data.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

// SymMat * SymMat
impl ops::Mul for SymMat {
    type Output = SymMat;
    fn mul(self, rhs: SymMat) -> SymMat {
        assert_eq!(self.cols, rhs.rows, "SymMat mul: dimension mismatch");
        let mut data = Vec::with_capacity(self.rows * rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum: Option<E> = None;
                for k in 0..self.cols {
                    let prod = self.get(i, k).clone() * rhs.get(k, j).clone();
                    sum = Some(match sum {
                        Some(acc) => acc + prod,
                        None => prod,
                    });
                }
                data.push(sum.unwrap_or_else(|| constant(0.0)));
            }
        }
        SymMat { rows: self.rows, cols: rhs.cols, data }
    }
}

// SymMat * SymVec
impl ops::Mul<SymVec> for SymMat {
    type Output = SymVec;
    fn mul(self, rhs: SymVec) -> SymVec {
        assert_eq!(self.cols, rhs.len(), "SymMat * SymVec: dimension mismatch");
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum: Option<E> = None;
            for j in 0..self.cols {
                let prod = self.get(i, j).clone() * rhs[j].clone();
                sum = Some(match sum {
                    Some(acc) => acc + prod,
                    None => prod,
                });
            }
            result.push(sum.unwrap_or_else(|| constant(0.0)));
        }
        SymVec(result)
    }
}

// SymMat * E (scalar)
impl ops::Mul<E> for SymMat {
    type Output = SymMat;
    fn mul(self, rhs: E) -> SymMat {
        SymMat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.into_iter().map(|e| e * rhs.clone()).collect(),
        }
    }
}

// E * SymMat (scalar)
impl ops::Mul<SymMat> for E {
    type Output = SymMat;
    fn mul(self, rhs: SymMat) -> SymMat {
        SymMat {
            rows: rhs.rows,
            cols: rhs.cols,
            data: rhs.data.into_iter().map(|e| self.clone() * e).collect(),
        }
    }
}

impl fmt::Display for SymMat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.rows {
            if i > 0 { write!(f, "; ")?; }
            for j in 0..self.cols {
                if j > 0 { write!(f, ", ")?; }
                fmt::Display::fmt(self.get(i, j), f)?;
            }
        }
        write!(f, "]")
    }
}

// ============================================================
// Jacobian
// ============================================================

/// Compute the Jacobian matrix: partial derivatives of each expression with
/// respect to each variable.
///
/// Returns a [`SymMat`] with `exprs.len()` rows and `vars.len()` columns,
/// where element (i, j) is `d(exprs[i]) / d(vars[j])`.
pub fn jacobian(exprs: &[E], vars: &[&str]) -> SymMat {
    let rows = exprs.len();
    let cols = vars.len();
    let mut data = Vec::with_capacity(rows * cols);
    for expr in exprs {
        for var in vars {
            data.push(expr.diff(var));
        }
    }
    SymMat { rows, cols, data }
}
