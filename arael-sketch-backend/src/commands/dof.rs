use super::*;

/// Classify free directions from DofResult eigenvectors.
pub(crate) fn classify_dof_directions(result: &arael_sketch_solver::DofResult) -> Vec<String> {
    let threshold = 1e-6;
    let n = result.eigenvalues.len();
    let mut free_dirs = Vec::new();
    for col in 0..n {
        if result.eigenvalues[col].abs() > threshold { continue; }
        let ev = &result.eigenvectors[col];

        let max_comp = ev.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        if max_comp < 1e-10 { continue; }
        let comp_threshold = max_comp * 0.1;

        let mut parts: Vec<(String, f64)> = Vec::new();
        for i in 0..n {
            if ev[i].abs() > comp_threshold {
                let name = if result.param_names[i].is_empty() {
                    format!("param[{}]", i)
                } else {
                    result.param_names[i].clone()
                };
                parts.push((name, ev[i]));
            }
        }
        if parts.is_empty() { continue; }
        free_dirs.push(classify_free_direction(&parts));
    }
    free_dirs
}

pub(crate) fn cmd_dof_eigenvalues(ctx: &mut CommandContext, raw: bool) -> CmdResult {
    let t0 = web_time::Instant::now();
    // Hessian-based diagnostic. Default is the symmetrically Jacobi-
    // preconditioned Hessian `D^{-1} H D^{-1}` with `D =
    // diag(sqrt(diag(H)))` -- the null-space is preserved (rank
    // unchanged) but the per-parameter scale differences that
    // otherwise make eigenvalue rank-detection break at high sketch
    // scales are folded into `D`. Eigenvectors back-transformed to
    // raw parameter space. `raw` shows the un-preconditioned Hessian
    // for residual-design debugging.
    let result = ctx.sketch.get_mut().compute_dof_eigenvalues_opt(true, !raw)?;
    let t_total = t0.elapsed();
    let n = result.eigenvalues.len();
    if n == 0 {
        return Ok(ok("Hessian: 0x0 (empty)".to_string()));
    }
    let header = if raw { "Hessian (raw)" } else { "Hessian (preconditioned)" };
    let mut lines = vec![format!("{}: {}x{}, DOF: {}, time: {:.2}ms",
        header, n, n, result.dof, t_total.as_secs_f64() * 1000.0)];
    let mut evs: Vec<(f64, usize)> = result.eigenvalues.iter().cloned().enumerate().map(|(i,v)| (v, i)).collect();
    evs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (val, col) in &evs {
        let ev = &result.eigenvectors[*col];
        let max_comp = ev.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let comp_threshold = max_comp * 0.3;
        // Render as linear combination: "+0.707 A0.start_angle -0.707 A0.end_angle"
        let parts: Vec<String> = (0..n).filter(|&i| ev[i].abs() > comp_threshold)
            .enumerate()
            .map(|(k, i)| {
                let name = if result.param_names[i].is_empty() { format!("[{}]", i) } else { result.param_names[i].clone() };
                let v = ev[i];
                if k == 0 {
                    if v < 0.0 { format!("-{:.3} {}", -v, name) } else { format!("{:.3} {}", v, name) }
                } else {
                    if v < 0.0 { format!("- {:.3} {}", -v, name) } else { format!("+ {:.3} {}", v, name) }
                }
            })
            .collect();
        lines.push(format!("  {:.6e}  {}", val, parts.join(" ")));
    }
    Ok(ok(lines.join("\n")))
}

pub(crate) fn cmd_dof_singular(ctx: &mut CommandContext, raw: bool) -> CmdResult {
    use arael_sketch_solver::SymbolBag;
    ctx.sketch.get_mut().prepare_expr_constraints();
    let saved_drift = ctx.sketch.drift_isigma;
    ctx.sketch.get_mut().drift_isigma = 0.0;
    let mut params = Vec::new();
    ctx.sketch.mutate_values(|s| s.serialize(&mut params));
    let n = params.len();
    let bag = SymbolBag::build(&ctx.sketch);
    let mut idx_to_name: Vec<String> = vec![String::new(); n];
    for (name, &idx) in &bag.param_indices {
        let i = idx as usize;
        if i < n && idx_to_name[i].is_empty() { idx_to_name[i] = name.clone(); }
    }
    let t0 = web_time::Instant::now();
    let jacobian = ctx.sketch.get_mut().calc_jacobian(&params);
    let t_build = t0.elapsed();
    ctx.sketch.get_mut().drift_isigma = saved_drift;
    let m = jacobian.num_residuals();
    if m == 0 || n == 0 {
        return Ok(ok(format!("Jacobian: {} residuals x {} params (empty)", m, n)));
    }
    // Degenerate geometry yields NaN; an SVD iterating on it never
    // converges. Same guard as compute_dof.
    if jacobian.rows.iter().any(|r| r.entries.iter().any(|&(_, v)| !v.is_finite())) {
        return Err("Jacobian contains non-finite values (degenerate geometry)".into());
    }
    // Raw SVD: un-normalised Jacobian; sigmas carry the real
    // per-parameter scale, useful for spotting residual-design issues.
    // Normalised SVD (the default): each column of J scaled by
    // 1 / col_L2_norm, so the spectrum reflects only row-space linear
    // dependence. Right singular vectors get back-transformed to raw
    // parameter space (v_raw[i] = v_norm[i] / col_norms[i], renormed
    // to unit length) so the printed eigenvector still describes a
    // direction in the user's parameter coordinates.
    let t1 = web_time::Instant::now();
    let (svd, col_norms) = if raw {
        (jacobian.svd(), Vec::new())
    } else {
        jacobian.svd_column_normalised()
    };
    let t_svd = t1.elapsed();
    let svs_vec = &svd.singular_values;
    let k_dim = svs_vec.len();
    // Re-pack V (n x k row-major) and U (m x k row-major) into accessors
    // that mirror the legacy nalgebra/faer API (sv at singular-value index
    // -> direction in parameter space for V, contribution per row for U).
    let v_row = |idx: usize| -> Vec<f64> {
        let mut row = vec![0.0f64; n];
        for i in 0..n { row[i] = svd.v[i * k_dim + idx]; }
        row
    };
    let u_col = |idx: usize| -> Vec<f64> {
        let mut col = vec![0.0f64; m];
        for i in 0..m { col[i] = svd.u[i * k_dim + idx]; }
        col
    };

    let labels = ctx.sketch.constraint_labels();

    let header = if raw { "Jacobian (raw)" } else { "Jacobian (column-normalised)" };
    let mut lines = vec![format!("{}: {} residuals x {} params", header, m, n)];
    lines.push(format!("  build: {:.2}ms, svd: {:.2}ms",
        t_build.as_secs_f64() * 1000.0,
        t_svd.as_secs_f64() * 1000.0));
    let mut svs: Vec<(f64, usize)> = svs_vec.iter().cloned().enumerate().map(|(i,v)| (v, i)).collect();
    svs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (val, idx) in &svs {
        // Right singular vector: direction in parameter space.
        // For the normalised SVD we back-transform by dividing each
        // component by the corresponding column norm and then rescaling
        // to unit length, so the printed eigenvector still describes a
        // physical direction the user can act on. Raw SVD needs no
        // transform.
        let sv_raw = v_row(*idx);
        let sv: Vec<f64> = if raw {
            sv_raw
        } else {
            let mut adjusted: Vec<f64> = sv_raw.iter().enumerate()
                .map(|(i, &v)| v / col_norms[i].max(1e-15))
                .collect();
            let norm: f64 = adjusted.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-20 {
                for v in adjusted.iter_mut() { *v /= norm; }
            }
            adjusted
        };
        let max_comp = sv.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let comp_threshold = max_comp * 0.3;
        let parts: Vec<String> = (0..n).filter(|&i| sv[i].abs() > comp_threshold)
            .enumerate()
            .map(|(k, i)| {
                let name = if idx_to_name[i].is_empty() { format!("[{}]", i) } else { idx_to_name[i].clone() };
                let v = sv[i];
                if k == 0 {
                    if v < 0.0 { format!("-{:.3} {}", -v, name) } else { format!("{:.3} {}", v, name) }
                } else {
                    if v < 0.0 { format!("- {:.3} {}", -v, name) } else { format!("+ {:.3} {}", v, name) }
                }
            })
            .collect();
        lines.push(format!("  {:.6e}  {}", val, parts.join(" ")));

        // Left singular vector: which constraints project onto this direction.
        // Aggregate u[row]^2 by (cid, label) — the label gives attribute-level
        // granularity (e.g. Arc:0 vs Arc:4) while cid gives the instance.
        let uv = u_col(*idx);
        let mut weight: std::collections::HashMap<(u32, &'static str), f64> = std::collections::HashMap::new();
        for (row_idx, row) in jacobian.rows.iter().enumerate() {
            if row_idx >= uv.len() { break; }
            let w = uv[row_idx];
            *weight.entry((row.constraint, row.label)).or_insert(0.0) += w * w;
        }
        // Sum over all constraints = 1 (u is a unit vector). Report as percentages.
        let mut contribs: Vec<((u32, &'static str), f64)> = weight.into_iter().collect();
        contribs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_max = contribs.first().map(|(_, w)| *w).unwrap_or(0.0);
        let top_threshold = top_max * 0.1; // show anything >= 10% of dominant
        contribs.retain(|(_, w)| *w > top_threshold);
        contribs.truncate(6);
        if !contribs.is_empty() {
            let parts: Vec<String> = contribs.iter().map(|((cid, label), w)| {
                // Instance label like "arc:A0" or "parallel:L3,L0"; take the part after ":"
                // for compactness since it already identifies the entity.
                let instance = labels.get(cid).cloned().unwrap_or_else(|| format!("cid={}", cid));
                let instance_short = instance.split_once(':').map(|x| x.1).unwrap_or(&instance).to_string();
                format!("{:.0}% {}/{}", w * 100.0, instance_short, label)
            }).collect();
            lines.push(format!("           {}", parts.join(", ")));
        }
    }
    Ok(ok(lines.join("\n")))
}

pub(crate) fn cmd_dof_jacobian(ctx: &mut CommandContext) -> CmdResult {
    use arael_sketch_solver::SymbolBag;
    ctx.sketch.get_mut().prepare_expr_constraints();
    let saved_drift = ctx.sketch.drift_isigma;
    ctx.sketch.get_mut().drift_isigma = 0.0;
    let mut params = Vec::new();
    ctx.sketch.mutate_values(|s| s.serialize(&mut params));
    let n = params.len();
    if n == 0 {
        ctx.sketch.get_mut().drift_isigma = saved_drift;
        return Ok(ok("No params".to_string()));
    }
    let bag = SymbolBag::build(&ctx.sketch);
    let mut idx_to_name: Vec<String> = vec![String::new(); n];
    for (name, &idx) in &bag.param_indices {
        let i = idx as usize;
        if i < n && idx_to_name[i].is_empty() { idx_to_name[i] = name.clone(); }
    }
    let jacobian = ctx.sketch.get_mut().calc_jacobian(&params);
    let labels = ctx.sketch.constraint_labels();
    ctx.sketch.get_mut().drift_isigma = saved_drift;
    let mut lines = vec![format!("Jacobian: {} rows x {} cols", jacobian.num_residuals(), n)];
    for (i, row) in jacobian.rows.iter().enumerate() {
        let entries: Vec<String> = row.entries.iter()
            .map(|&(idx, val)| {
                let name = if idx_to_name[idx as usize].is_empty() {
                    format!("[{}]", idx)
                } else {
                    idx_to_name[idx as usize].clone()
                };
                // Collapse exact zeros to "<name>=0" to cut noise while keeping
                // the parameter visible as part of the constraint.
                if val == 0.0 {
                    format!("{}=0", name)
                } else {
                    format!("{}={:+.6}", name, val)
                }
            })
            .collect();
        let norm: f64 = row.entries.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
        // Combine instance (from cid->label map) with attribute label (from row.label).
        let instance = labels.get(&row.constraint).cloned().unwrap_or_else(|| format!("cid={}", row.constraint));
        let instance_short = instance.split_once(':').map(|x| x.1).unwrap_or(&instance).to_string();
        let combined = format!("{}/{}", instance_short, row.label);
        let r_str = if row.residual == 0.0 { "r=0".to_string() } else { format!("r={:+.6e}", row.residual) };
        let norm_str = if norm == 0.0 { "|dr|=0".to_string() } else { format!("|dr|={:.6e}", norm) };
        lines.push(format!("  row {:3} cid={:3} {:30} {:16} {:16} dr/d[{}]",
            i, row.constraint, combined, r_str, norm_str, entries.join(", ")));
    }
    Ok(ok(lines.join("\n")))
}

pub(crate) fn cmd_dof(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let arg = args.trim();
    if arg == "eigenvalues" {
        return cmd_dof_eigenvalues(ctx, false);
    }
    if arg == "eigenvalues raw" {
        return cmd_dof_eigenvalues(ctx, true);
    }
    // `dof singular` uses the column-normalised Jacobian (each column
    // scaled by 1 / L2_norm) so the sigma spectrum reflects row-space
    // linear dependence only, not per-parameter scale conditioning.
    // This matches what the internal rank detector uses. `dof singular
    // raw` falls back to the un-normalised Jacobian, useful when
    // debugging residual scaling choices where the raw sigmas carry
    // meaningful physical magnitudes.
    if arg == "singular" {
        return cmd_dof_singular(ctx, false);
    }
    if arg == "singular raw" {
        return cmd_dof_singular(ctx, true);
    }
    if arg == "jacobian" {
        return cmd_dof_jacobian(ctx);
    }
    if !arg.is_empty() && arg != "analyze" {
        return Err("Usage: dof | dof analyze | dof eigenvalues [raw] | dof singular [raw] | dof jacobian".into());
    }

    if arg == "analyze" {
        // The eigenvector analysis has no cached form; it is a
        // diagnostic and may bump the generation.
        let result = ctx.sketch.get_mut().compute_dof(true)?;
        let free_dirs = classify_dof_directions(&result);
        let mut lines = vec![format!("DOF: {}", result.dof)];
        for (i, desc) in free_dirs.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, desc));
        }
        Ok(ok(lines.join("\n")))
    } else {
        // Plain count: the read door, so the cache and the warm
        // session survive the query.
        match ctx.sketch.dof() {
            Ok(d) => Ok(ok(format!("DOF: {}", d))),
            Err(e) => Err(e.into()),
        }
    }
}

/// Classify a free direction from its eigenvector components.
pub(crate) fn classify_free_direction(parts: &[(String, f64)]) -> String {
    // Single parameter free
    if parts.len() == 1 {
        return format!("{} is free", parts[0].0);
    }

    // Collect entity names and check motion patterns
    let mut entities = std::collections::BTreeSet::new();
    let mut all_x = true;
    let mut all_y = true;
    let mut has_non_xy = false;

    for (name, _val) in parts {
        // Extract entity name (e.g., "L0" from "L0.p1.x")
        let entity = name.split('.').next().unwrap_or(name);
        entities.insert(entity.to_string());

        if name.ends_with(".x") {
            all_y = false;
        } else if name.ends_with(".y") {
            all_x = false;
        } else {
            // radius, angle, etc.
            all_x = false;
            all_y = false;
            has_non_xy = true;
        }
    }

    let entity_list: Vec<&str> = entities.iter().map(|s| s.as_str()).collect();
    let entity_str = if entity_list.len() <= 9 {
        entity_list.join(", ")
    } else {
        format!("{} entities", entity_list.len())
    };

    // Check for pure translation
    if all_x && !has_non_xy {
        return format!("translate X: {}", entity_str);
    }
    if all_y && !has_non_xy {
        return format!("translate Y: {}", entity_str);
    }

    // Check for uniform translation (all x components equal AND all y components equal)
    let x_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".x"))
        .map(|(_, v)| *v).collect();
    let y_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".y"))
        .map(|(_, v)| *v).collect();

    if !has_non_xy && !x_vals.is_empty() && !y_vals.is_empty()
        && y_vals.iter().all(|v| v.abs() < 1e-6) {
        // All Y near zero, only X moves
        return format!("translate X: {}", entity_str);
    }
    if !has_non_xy && !x_vals.is_empty() && !y_vals.is_empty()
        && x_vals.iter().all(|v| v.abs() < 1e-6) {
        return format!("translate Y: {}", entity_str);
    }

    // Check for rotation: x and y components should follow tangent pattern
    // For rotation around centroid, dx_i ~ -(y_i - cy), dy_i ~ (x_i - cx)
    // Simplified: if x and y components are mixed and entities share the motion, call it rotation
    if !x_vals.is_empty() && !y_vals.is_empty() && x_vals.len() == y_vals.len() && !has_non_xy {
        // Check if all translation components are equal (pure translation)
        let all_x_equal = x_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        let all_y_equal = y_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        if all_x_equal && all_y_equal {
            return format!("translate: {}", entity_str);
        }
        return format!("rotate: {}", entity_str);
    }

    // Check for single-entity multi-param freedom
    if entities.len() == 1 {
        let param_list: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        return format!("{} free: {}", entity_list[0], param_list.join(", "));
    }

    // Fallback: list participating entities
    format!("coupled motion: {}", entity_str)
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

