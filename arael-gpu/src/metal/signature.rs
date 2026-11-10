//! Kernel entries rewritten for Metal: Metal takes a kernel's arguments as
//! buffer bindings (`[[buffer(i)]]`) and its thread positions as attributed
//! parameters, where CUDA and HIP take plain parameters and builtins.
//!
//! An entry is a line that starts with `KERNEL...` (or a name ending in
//! `_ENTRY`, a macro standing for one), then `void name(params) {`. A
//! parameter is a buffer, `GLOBAL [const] T* [RESTRICT] name`, or a scalar,
//! `T name`. Each becomes a binding, and the body starts by copying the
//! bindings into locals of the written names and types, so a body that
//! assigns to a parameter works as with a by-value argument:
//!
//! ```text
//! KERNEL_THREADS(256) void k(GLOBAL float* RESTRICT x, u32 n) {
//! ```
//!
//! becomes
//!
//! ```text
//! KERNEL_THREADS(256) void k(GLOBAL float* arael_a0 [[buffer(0)]], constant u32& arael_a1 [[buffer(1)]],
//!     uint3 arael_tid [[thread_position_in_threadgroup]], ...) { GLOBAL float* RESTRICT x = arael_a0; u32 n = arael_a1;
//! ```
//!
//! arael_gpu.h's Metal branch reads `arael_tid`, `arael_bid` and `arael_bdim`
//! in `thread_x()` and its kin, and `arael_dyn` (added when the body uses
//! `DYNAMIC_SHARED`) in `DYNAMIC_SHARED`. The rewritten parameters keep the
//! entry's line count, so the compiler's line numbers match the source.

/// A parameter of a kernel entry, as a launch binds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Param {
    /// A pointer: `setBuffer:offset:atIndex:`.
    Buffer,
    /// A value of the written type: `setBytes:length:atIndex:`.
    Scalar(String),
}

/// A kernel entry of a source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub name: String,
    pub params: Vec<Param>,
    /// Whether the body uses `DYNAMIC_SHARED`: the launch's dynamic shared
    /// memory is bound at threadgroup index 0.
    pub dynamic_shared: bool,
}

/// `source` with every kernel entry's parameters as Metal bindings, and the
/// entries in source order; `Err` names an entry the rewrite cannot read.
pub(crate) fn rewrite(source: &str) -> Result<(String, Vec<Entry>), String> {
    let b = source.as_bytes();
    let mut out = String::with_capacity(source.len() + source.len() / 8);
    let mut entries = Vec::new();
    // `at` is the start of a line; `copied` how far `source` is in `out`
    let (mut at, mut copied) = (0, 0);
    while at < b.len() {
        if let Some((name, open)) = entry_head(b, at) {
            let close = matching(b, open, b'(', b')').ok_or_else(|| format!("kernel `{name}`: no `)` closes its parameters"))?;
            let brace = skip_space(b, close + 1);
            if b.get(brace) != Some(&b'{') {
                return Err(format!("kernel `{name}`: its parameters are not followed by its body"));
            }
            let end = matching(b, brace, b'{', b'}').ok_or_else(|| format!("kernel `{name}`: no `}}` closes its body"))?;
            let dynamic_shared = source[brace..end].contains("DYNAMIC_SHARED(");
            let written = &source[open + 1..close];
            let (bindings, prologue, params) = parameters(&name, written)?;
            let mut list = bindings;
            list.push("uint3 arael_tid [[thread_position_in_threadgroup]]".to_string());
            list.push("uint3 arael_bid [[threadgroup_position_in_grid]]".to_string());
            list.push("uint3 arael_bdim [[threads_per_threadgroup]]".to_string());
            if dynamic_shared {
                list.push("threadgroup u8* arael_dyn [[threadgroup(0)]]".to_string());
            }
            out.push_str(&source[copied..open + 1]);
            out.push_str(&list.join(", "));
            out.push(')');
            // the lines the written parameters took, so later lines keep their numbers
            out.extend(core::iter::repeat_n('\n', written.matches('\n').count()));
            out.push_str(" {");
            out.push_str(&prologue);
            copied = brace + 1;
            entries.push(Entry { name, params, dynamic_shared });
            // on from the line of the body's `{`
            at = brace;
        }
        at = match source[at..].find('\n') {
            Some(n) => at + n + 1,
            None => b.len(),
        };
    }
    out.push_str(&source[copied..]);
    Ok((out, entries))
}

/// The entry starting the line at `at`: its name and the byte of the `(`
/// opening its parameters.
fn entry_head(b: &[u8], at: usize) -> Option<(String, usize)> {
    let word_end = |from: usize| (from..b.len()).find(|&i| !(b[i].is_ascii_alphanumeric() || b[i] == b'_')).unwrap_or(b.len());
    let head_end = word_end(at);
    let head = core::str::from_utf8(&b[at..head_end]).ok()?;
    if !(head.starts_with("KERNEL") || head.ends_with("_ENTRY")) {
        return None;
    }
    let mut i = head_end;
    // the entry macro's own arguments, `KERNEL_THREADS(n)`
    if b.get(i) == Some(&b'(') {
        i = matching(b, i, b'(', b')')? + 1;
    }
    i = skip_space(b, i);
    let void_end = word_end(i);
    if &b[i..void_end] != b"void" {
        return None;
    }
    i = skip_space(b, void_end);
    let name_end = word_end(i);
    if name_end == i {
        return None;
    }
    let name = core::str::from_utf8(&b[i..name_end]).ok()?.to_string();
    let open = skip_space(b, name_end);
    (b.get(open) == Some(&b'(')).then_some((name, open))
}

/// The first byte at or after `i` that is not white space.
fn skip_space(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// The byte of the `close` that matches the `open` at `at`, comments
/// skipped.
fn matching(b: &[u8], at: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = at;
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
                continue;
            }
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The bindings, the prologue and the kinds of the parameters `written` of
/// the entry `name`.
fn parameters(name: &str, written: &str) -> Result<(Vec<String>, String, Vec<Param>), String> {
    let mut bindings = Vec::new();
    let mut prologue = String::new();
    let mut params = Vec::new();
    let list = written.trim();
    if list.is_empty() || list == "void" {
        return Ok((bindings, prologue, params));
    }
    for (i, p) in list.split(',').enumerate() {
        let p = p.split_whitespace().collect::<Vec<_>>().join(" ");
        let cut = p.rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |k| k + 1);
        let (ty, var) = (p[..cut].trim(), &p[cut..]);
        if ty.is_empty() || var.is_empty() || var.as_bytes()[0].is_ascii_digit() {
            return Err(format!("kernel `{name}`: parameter {i} (`{p}`) is not `type name`"));
        }
        if ty.contains('&') || ty.contains('[') {
            return Err(format!("kernel `{name}`: parameter {i} (`{p}`) is neither a pointer nor a value"));
        }
        if ty.contains('*') {
            let words: Vec<&str> = ty.split(|c: char| c.is_whitespace() || c == '*').filter(|w| !w.is_empty()).collect();
            if !words.iter().any(|w| *w == "GLOBAL" || *w == "CONSTANT") {
                return Err(format!("kernel `{name}`: parameter {i} (`{p}`) is a pointer without GLOBAL or CONSTANT"));
            }
            let bound = ty.replace("RESTRICT", "").split_whitespace().collect::<Vec<_>>().join(" ");
            bindings.push(format!("{bound} arael_a{i} [[buffer({i})]]"));
            params.push(Param::Buffer);
        } else {
            bindings.push(format!("constant {ty}& arael_a{i} [[buffer({i})]]"));
            params.push(Param::Scalar(ty.to_string()));
        }
        prologue.push_str(&format!(" {ty} {var} = arael_a{i};"));
    }
    Ok((bindings, prologue, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_and_scalars_become_bindings_and_locals() {
        let src = "// k\nKERNEL_THREADS(256) void k(GLOBAL const float* RESTRICT x, GLOBAL W* y, u32 n) {\n    y[0] = x[n];\n}\n";
        let (out, entries) = rewrite(src).unwrap();
        assert_eq!(
            out,
            "// k\nKERNEL_THREADS(256) void k(GLOBAL const float* arael_a0 [[buffer(0)]], GLOBAL W* arael_a1 [[buffer(1)]], \
             constant u32& arael_a2 [[buffer(2)]], uint3 arael_tid [[thread_position_in_threadgroup]], \
             uint3 arael_bid [[threadgroup_position_in_grid]], uint3 arael_bdim [[threads_per_threadgroup]]) { \
             GLOBAL const float* RESTRICT x = arael_a0; GLOBAL W* y = arael_a1; u32 n = arael_a2;\n    y[0] = x[n];\n}\n"
        );
        assert_eq!(entries, vec![Entry { name: "k".into(), params: vec![Param::Buffer, Param::Buffer, Param::Scalar("u32".into())], dynamic_shared: false }]);
    }

    #[test]
    fn every_entry_form_is_found_and_the_rest_left_alone() {
        let src = "#define SYRK_ENTRY KERNEL_THREADS(32)\nDEVICE void helper(GLOBAL float* x) {}\nKERNEL void a() {}\n\
                   SYRK_ENTRY void b(u32 n) { DYNAMIC_SHARED(float, s); }\nKERNEL_THREADS_BLOCKS(THREADS, 2) void c(GLOBAL u32* f) {}\n";
        let (out, entries) = rewrite(src).unwrap();
        let names: Vec<_> = entries.iter().map(|e| (e.name.as_str(), e.dynamic_shared)).collect();
        assert_eq!(names, [("a", false), ("b", true), ("c", false)]);
        assert!(out.starts_with("#define SYRK_ENTRY KERNEL_THREADS(32)\nDEVICE void helper(GLOBAL float* x) {}\n"));
        assert!(out.contains("threadgroup u8* arael_dyn [[threadgroup(0)]]) { u32 n = arael_a0; DYNAMIC_SHARED(float, s); }"));
    }

    #[test]
    fn parameters_over_several_lines_keep_the_line_count() {
        let src = "KERNEL void k(GLOBAL float* x,\n              u32 n) {\n  x[0] = 1.0f; // }\n}\nKERNEL void j() {}\n";
        let (out, entries) = rewrite(src).unwrap();
        assert_eq!(out.matches('\n').count(), src.matches('\n').count());
        assert_eq!(entries.len(), 2);
        assert!(out.ends_with("KERNEL void j(uint3 arael_tid [[thread_position_in_threadgroup]], uint3 arael_bid [[threadgroup_position_in_grid]], uint3 arael_bdim [[threads_per_threadgroup]]) {}\n"));
    }

    #[test]
    fn what_the_rewrite_cannot_read_is_an_error_naming_the_entry() {
        let err = |s: &str| rewrite(s).unwrap_err();
        assert!(err("KERNEL void k(float* x) {}").contains("`k`: parameter 0 (`float* x`) is a pointer without GLOBAL"));
        assert!(err("KERNEL void k(u32& n) {}").contains("`k`: parameter 0"));
        assert!(err("KERNEL void k(GLOBAL float* x);").contains("`k`: its parameters are not followed by its body"));
        assert!(err("KERNEL void k(GLOBAL float* x) { x[0] = 1;").contains("`k`: no `}` closes its body"));
    }
}
