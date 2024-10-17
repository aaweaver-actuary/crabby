/// Enumeration representing the transpose operation for BLAS
#[derive(Debug, PartialEq)]
pub enum BlasTransposeFlag {
    NoTranspose,
    Transpose,
    ConjugateTranspose,
}

impl BlasTransposeFlag {
    pub fn to_blas_char(&self) -> u8 {
        match self {
            BlasTransposeFlag::NoTranspose => b'N',
            BlasTransposeFlag::Transpose => b'T',
            BlasTransposeFlag::ConjugateTranspose => b'C',
        }
    }
}
