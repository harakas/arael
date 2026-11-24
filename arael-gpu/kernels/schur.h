// schur.h: what the kernels over a Schur reduction's plan share: the plan's
// header, the reduction's scalar R, the sizes words of an observer and of a
// kept block's tile entry, an observer's coupling tile and the solve with
// an eliminated block's factor. Compiled before schur.cpp (the reduction
// and its refinement) and before cg.cpp (conjugate gradients on the reduced
// system), after wide.h, with schur.rs's #defines (schur.cpp lists them).

// The plan header: the word offset of each of the plan's arrays, a slot
// each, a launch's parameter (TABLE_PARAM(SchurHeader) H, read from the
// launch's parameters, not from the plan) in every kernel that takes the
// plan.
struct SchurHeader {
    u32 w[SHEADER];
};

// One plane: the threads of the tile, rhs and symv kernels.
#define PLANE 32u
#if PLANE != WARP
#error "schur.h: the rhs and symv kernels sum over a plane with warp_sum"
#endif
// No kept-kept tile of H behind a tile of S.
#define NONE 0xffffffffu

// A kept block's rows as lanes. The kernels over a kept block's items (its
// observers, its tiles of S) run a block of warps per kept block, each
// warp MGROUPS groups of KW lanes: a lane is a row of the block, and a
// group takes an item at a time, so its lanes read neighbouring values at
// every step. For kept blocks no wider than a plane has lanes; wider ones
// keep one warp a block with its lanes over the items.
#if KW <= PLANE
#define MGROUPS (PLANE / KW)
#endif

// R: the scalar the reduction computes in. W itself with SCHUR_F32 0; f32
// with SCHUR_F32 1, as the host's f32 solve, for a solve whose factor is f32
// and that does not refine against S: the inputs (H, the gradient, the kept
// blocks' solution) are rounded to f32 as read (an f32 root's H and
// gradient already are), and the eliminated blocks' factors, z, W, S, the
// reduced right-hand side and the kept blocks' solution are kept in f32.
#if SCHUR_F32
typedef float R;
#define R_DF32 0
DEVICE R r_fma(R acc, R x, R y) {
    return fma(x, y, acc);
}
#else
typedef W R;
#define R_DF32 ARAEL_WIDE_DF32
DEVICE R r_fma(R acc, R x, R y) {
    return wide_fma(acc, x, y);
}
#endif
// v in R: a value of H or the gradient (V), or of a vector (W).
template <typename T> DEVICE R to_r(T v) {
    return (R)v;
}

// An observer's sizes word (SO_DIMS; schur.rs, so_dims): the width of its
// kept block, of its eliminated block, and whether its coupling tile is
// stored (eliminated, kept).
#define DIMS_WO(d) ((d) & 0xffu)
#define DIMS_WE(d) (((d) >> 8) & 0xffu)
#define DIMS_TRANS(d) ((d) >> 16)

// A kept block's tile entry's sizes word (KT_DIMS; schur.rs, kt_dims): the
// tile's rows and columns (at most 64), whether the block is its column
// block, whether it is a diagonal tile.
#define KT_PACK(wr, wc, trans, diag) ((wr) | (wc) << 8 | (trans) << 16 | (diag) << 17)
#define KT_WR(d) ((d) & 0xffu)
#define KT_WC(d) (((d) >> 8) & 0xffu)
#define KT_TRANS(d) (((d) >> 16) & 1u)
#define KT_DIAG(d) ((d) >> 17)

// C[i][k] of an observer's coupling tile at `ca`, wo x we, in either storage
// orientation.
DEVICE V coupling(GLOBAL const V* h, u32 ca, u32 trans, u32 wo, u32 we, u32 i, u32 k) {
    return trans != 0 ? h[ca + k + i * we] : h[ca + i + k * wo];
}

// v = D^-1 v in place (w <= EW entries) by two triangular solves with D's
// factor at `f`, column-major w x w: D = U P U^T, U unit lower (D's
// Cholesky factor L with each column divided by its diagonal entry) below
// the diagonal, P = diag(L_jj^2) as its reciprocals on the diagonal. The
// host's llt_solve_panel up to rounding, at one product a term: solves keep
// S the reduction of a nearby positive definite system where multiplying
// by D's rounded inverse does not (its error grows with D's condition
// number, high for a point seen along nearly parallel rays). wide_fma's
// rounding, relative to its terms, keeps the solves backward stable.
DEVICE void ldlt_solve(GLOBAL const R* f, u32 w, THREAD R* v) {
#pragma unroll
    for (u32 k = 0; k < EW; k++) {
        if (k < w) {
            R s = v[k];
#pragma unroll
            for (u32 j = 0; j < k; j++) s = r_fma(s, -f[k + j * w], v[j]);
            v[k] = s;
        }
    }
#pragma unroll
    for (u32 k = 0; k < EW; k++)
        if (k < w) v[k] = v[k] * f[k + k * w];
#pragma unroll
    for (u32 kk = 0; kk < EW; kk++) {
        u32 k = EW - 1 - kk;
        if (k < w) {
            R s = v[k];
#pragma unroll
            for (u32 j = k + 1; j < EW; j++)
                if (j < w) s = r_fma(s, -f[j + k * w], v[j]);
            v[k] = s;
        }
    }
}
