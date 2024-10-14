use std::fmt;

#[derive(Debug)]
pub enum LinearAlgebraError {
    QrDecompositionError(String),
    MatrixInverseError(String),
    DotProductError(String),
}

impl fmt::Display for LinearAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LinearAlgebraError::QrDecompositionError(ref err) => {
                write!(f, "QR Decomposition Error: {}", err)
            }
            LinearAlgebraError::MatrixInverseError(ref err) => {
                write!(f, "Matrix Inverse Error: {}", err)
            }
            LinearAlgebraError::DotProductError(ref err) => write!(f, "Dot Product Error: {}", err),
        }
    }
}
