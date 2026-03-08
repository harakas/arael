// Eigen sparse Cholesky bridge for arael.
//
// Provides extern "C" functions wrapping:
// - Eigen::SimplicialLLT (f32 and f64)
// - Eigen::CholmodSimplicialLLT (f64 only, guarded by ARAEL_CHOLMOD define)
//
// CSC format: upper triangle, col_ptr as int64_t (Rust usize), row_idx as int32_t (Rust u32).
// Eigen uses int (32-bit) StorageIndex by default; col_ptr is narrowed once and cached.

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cstdint>
#include <vector>

#ifdef ARAEL_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

// -- Templated SimplicialLLT handle --

template<typename Scalar>
struct LLTHandle {
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    using Solver = Eigen::SimplicialLLT<SpMat, Eigen::Upper>;

    Solver solver;
    bool pattern_analyzed;
    std::vector<int> col_ptr_i32;

    LLTHandle() : pattern_analyzed(false) {}
};

template<typename Scalar>
static bool llt_solve(
    void* handle_ptr,
    int n, int nnz,
    const int64_t* col_ptr,
    const int32_t* row_idx,
    const Scalar* vals,
    const Scalar* rhs,
    Scalar* solution)
{
    try {
        auto* h = static_cast<LLTHandle<Scalar>*>(handle_ptr);
        using SpMat = typename LLTHandle<Scalar>::SpMat;

        // Convert col_ptr from int64 to int32 (cached allocation)
        h->col_ptr_i32.resize(n + 1);
        for (int i = 0; i <= n; i++) {
            h->col_ptr_i32[i] = static_cast<int>(col_ptr[i]);
        }

        // Map CSC data into Eigen sparse matrix (no copy of vals/row_idx)
        Eigen::Map<const SpMat> mat(
            n, n, nnz,
            h->col_ptr_i32.data(),
            reinterpret_cast<const int*>(row_idx),
            vals);

        if (!h->pattern_analyzed) {
            h->solver.analyzePattern(mat);
            if (h->solver.info() != Eigen::Success) return false;
            h->pattern_analyzed = true;
        }

        h->solver.factorize(mat);
        if (h->solver.info() != Eigen::Success) return false;

        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> b(rhs, n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = h->solver.solve(b);
        if (h->solver.info() != Eigen::Success) return false;
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(solution, n) = x;

        return true;
    } catch (...) {
        return false;
    }
}

// -- CholmodSimplicialLLT handle (f64 only) --

#ifdef ARAEL_CHOLMOD
struct CholmodHandle {
    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using Solver = Eigen::CholmodSimplicialLLT<SpMat, Eigen::Upper>;

    Solver solver;
    bool pattern_analyzed;
    std::vector<int> col_ptr_i32;

    CholmodHandle() : pattern_analyzed(false) {}
};

static bool cholmod_solve_impl(
    void* handle_ptr,
    int n, int nnz,
    const int64_t* col_ptr,
    const int32_t* row_idx,
    const double* vals,
    const double* rhs,
    double* solution)
{
    try {
        auto* h = static_cast<CholmodHandle*>(handle_ptr);
        using SpMat = CholmodHandle::SpMat;

        h->col_ptr_i32.resize(n + 1);
        for (int i = 0; i <= n; i++) {
            h->col_ptr_i32[i] = static_cast<int>(col_ptr[i]);
        }

        Eigen::Map<const SpMat> mat(
            n, n, nnz,
            h->col_ptr_i32.data(),
            reinterpret_cast<const int*>(row_idx),
            vals);

        if (!h->pattern_analyzed) {
            h->solver.analyzePattern(mat);
            h->pattern_analyzed = true;
        }

        h->solver.factorize(mat);
        if (h->solver.info() != Eigen::Success) return false;

        Eigen::Map<const Eigen::VectorXd> b(rhs, n);
        Eigen::Map<Eigen::VectorXd> x(solution, n);
        x = h->solver.solve(b);

        return h->solver.info() == Eigen::Success;
    } catch (...) {
        return false;
    }
}
#endif

// -- extern "C" interface --

extern "C" {

// f64 SimplicialLLT
void* eigen_llt_f64_create() { return new LLTHandle<double>(); }
void eigen_llt_f64_destroy(void* h) { delete static_cast<LLTHandle<double>*>(h); }
bool eigen_llt_f64_solve(void* h, int n, int nnz,
    const int64_t* cp, const int32_t* ri, const double* v,
    const double* rhs, double* sol) {
    return llt_solve<double>(h, n, nnz, cp, ri, v, rhs, sol);
}

// f32 SimplicialLLT
void* eigen_llt_f32_create() { return new LLTHandle<float>(); }
void eigen_llt_f32_destroy(void* h) { delete static_cast<LLTHandle<float>*>(h); }
bool eigen_llt_f32_solve(void* h, int n, int nnz,
    const int64_t* cp, const int32_t* ri, const float* v,
    const float* rhs, float* sol) {
    return llt_solve<float>(h, n, nnz, cp, ri, v, rhs, sol);
}

// f64 CholmodSimplicialLLT
#ifdef ARAEL_CHOLMOD
void* eigen_cholmod_f64_create() { return new CholmodHandle(); }
void eigen_cholmod_f64_destroy(void* h) { delete static_cast<CholmodHandle*>(h); }
bool eigen_cholmod_f64_solve(void* h, int n, int nnz,
    const int64_t* cp, const int32_t* ri, const double* v,
    const double* rhs, double* sol) {
    return cholmod_solve_impl(h, n, nnz, cp, ri, v, rhs, sol);
}
#endif

} // extern "C"
