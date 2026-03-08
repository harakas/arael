fn main() {
    #[cfg(feature = "lapack")]
    {
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=blas");
    }

    #[cfg(feature = "eigen")]
    {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("cpp/eigen_sparse.cpp")
            .include("/usr/include/eigen3")
            .flag("-std=c++17")
            .flag("-O2");

        #[cfg(feature = "cholmod")]
        {
            build.include("/usr/include/suitesparse");
            build.define("ARAEL_CHOLMOD", None);
        }

        build.compile("eigen_sparse");

        #[cfg(feature = "cholmod")]
        {
            println!("cargo:rustc-link-lib=cholmod");
            println!("cargo:rustc-link-lib=suitesparseconfig");
        }

        println!("cargo:rerun-if-changed=cpp/eigen_sparse.cpp");
    }
}
