use std::fmt;

#[derive(Debug)]
pub enum LinearAlgebraError {
    QrDecompositionError(String),
    MatrixInverseError(String),
    DotProductError(String),
    DimensionMismatchError(String),
    OperationFailedError(String),
    LapackError(String),
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
            LinearAlgebraError::DimensionMismatchError(ref err) => {
                write!(f, "Dimension Mismatch Error: {}", err)
            }
            LinearAlgebraError::OperationFailedError(ref err) => {
                write!(f, "Operation Failed Error: {}", err)
            }
            LinearAlgebraError::LapackError(ref err) => write!(f, "LAPACK Error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_decomposition_error_display() {
        let error =
            LinearAlgebraError::QrDecompositionError("Failed to decompose matrix".to_string());
        assert_eq!(
            format!("{}", error),
            "QR Decomposition Error: Failed to decompose matrix"
        );
    }

    #[test]
    fn test_matrix_inverse_error_display() {
        let error = LinearAlgebraError::MatrixInverseError("Matrix is singular".to_string());
        assert_eq!(
            format!("{}", error),
            "Matrix Inverse Error: Matrix is singular"
        );
    }

    #[test]
    fn test_dot_product_error_display() {
        let error =
            LinearAlgebraError::DotProductError("Vectors have different lengths".to_string());
        assert_eq!(
            format!("{}", error),
            "Dot Product Error: Vectors have different lengths"
        );
    }

    #[test]
    fn test_dimension_mismatch_error_display() {
        let error = LinearAlgebraError::DimensionMismatchError(
            "Matrix dimensions do not match".to_string(),
        );
        assert_eq!(
            format!("{}", error),
            "Dimension Mismatch Error: Matrix dimensions do not match"
        );
    }

    #[test]
    fn test_operation_failed_error_display() {
        let error = LinearAlgebraError::OperationFailedError(
            "Operation could not be completed".to_string(),
        );
        assert_eq!(
            format!("{}", error),
            "Operation Failed Error: Operation could not be completed"
        );
    }

    #[test]
    fn test_lapack_error_display() {
        let error = LinearAlgebraError::LapackError("LAPACK routine failed".to_string());
        assert_eq!(format!("{}", error), "LAPACK Error: LAPACK routine failed");
    }
}
