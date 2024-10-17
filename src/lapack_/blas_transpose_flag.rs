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

#[cfg(test)]

mod test {
    use super::*;

    #[test]
    fn test_blas_transpose_char() {
        let ch = BlasTransposeFlag::Transpose;
        assert_eq!(ch.to_blas_char(), b'T');
    }

    #[test]
    fn test_blas_conjugate_transpose_char() {
        let ch = BlasTransposeFlag::ConjugateTranspose;
        assert_eq!(ch.to_blas_char(), b'C');
    }
}
