fn main() {
    // Link against the LAPACK library
    println!("cargo:rustc-link-lib=dylib=lapack");
}
