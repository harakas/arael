fn main() {
    // `gpu_metal`: the Metal runtime is built, feature `gpu-metal` on an
    // Apple target (elsewhere there is no Metal to build).
    println!("cargo::rustc-check-cfg=cfg(gpu_metal)");
    if std::env::var_os("CARGO_FEATURE_GPU_METAL").is_some() && std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-cfg=gpu_metal");
    }

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

        // WARNING: CHOLMOD's Supernodal module is GPL-licensed. Enabling
        // `cholmod-gpl` makes the resulting binary subject to the GPL.
        #[cfg(feature = "cholmod-gpl")]
        {
            build.define("ARAEL_CHOLMOD_GPL", None);
            println!("cargo:warning=feature `cholmod-gpl` links CHOLMOD's GPL-licensed Supernodal module; the resulting binary is subject to the GPL");
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
