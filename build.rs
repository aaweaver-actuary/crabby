fn main() {
    println!("cargo:rustc-link-lib=dylib=openblas"); // Link OpenBLAS library
    println!("cargo:rustc-link-lib=dylib=blas"); // Link BLAS library
    println!("cargo:rustc-link-lib=dylib=lapack"); // Link LAPACK library

    // Add the library path
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
}
