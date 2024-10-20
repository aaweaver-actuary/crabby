use crate::errors::LinearAlgebraError;
use std::fmt;

#[derive(Debug)]
pub enum FittingError {
    QrDecompositionCalculationError,
    ModelFitError,
}

impl fmt::Display for FittingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FittingError::QrDecompositionCalculationError => {
                write!(f, "QR Decomposition Calculation Error")
            }
            FittingError::ModelFitError => write!(f, "Model Fit Error"),
        }
    }
}

impl From<LinearAlgebraError> for FittingError {
    fn from(error: LinearAlgebraError) -> Self {
        match error {
            LinearAlgebraError::QrDecompositionError(_err) => {
                FittingError::QrDecompositionCalculationError
            }
            LinearAlgebraError::MatrixInverseError(_err) => FittingError::ModelFitError,
            LinearAlgebraError::DotProductError(_err) => FittingError::ModelFitError,
            LinearAlgebraError::DimensionMismatchError(_err) => FittingError::ModelFitError,
            LinearAlgebraError::OperationFailedError(_err) => FittingError::ModelFitError,
            LinearAlgebraError::LapackError(_err) => FittingError::ModelFitError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::LinearAlgebraError;

    #[test]
    fn test_from_qr_decomposition_error() {
        let error = LinearAlgebraError::QrDecompositionError("QR Decomposition Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_qr_decomposition_error =
            matches!(fitting_error, FittingError::QrDecompositionCalculationError);
        assert!(fit_error_is_qr_decomposition_error);
    }

    #[test]
    fn test_from_matrix_inverse_error() {
        let error = LinearAlgebraError::MatrixInverseError("Matrix Inverse Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_model_fit_error = matches!(fitting_error, FittingError::ModelFitError);
        assert!(fit_error_is_model_fit_error);
    }

    #[test]
    fn test_from_dot_product_error() {
        let error = LinearAlgebraError::DotProductError("Dot Product Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_model_fit_error = matches!(fitting_error, FittingError::ModelFitError);
        assert!(fit_error_is_model_fit_error);
    }

    #[test]
    fn test_from_dimension_mismatch_error() {
        let error =
            LinearAlgebraError::DimensionMismatchError("Dimension Mismatch Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_model_fit_error = matches!(fitting_error, FittingError::ModelFitError);
        assert!(fit_error_is_model_fit_error);
    }

    #[test]
    fn test_from_operation_failed_error() {
        let error = LinearAlgebraError::OperationFailedError("Operation Failed Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_model_fit_error = matches!(fitting_error, FittingError::ModelFitError);
        assert!(fit_error_is_model_fit_error);
    }

    #[test]
    fn test_from_lapack_error() {
        let error = LinearAlgebraError::LapackError("LAPACK Error".to_string());
        let fitting_error: FittingError = error.into();
        let fit_error_is_model_fit_error = matches!(fitting_error, FittingError::ModelFitError);
        assert!(fit_error_is_model_fit_error);
    }

    #[test]
    fn test_display_qr_decomposition_calculation_error() {
        let error = FittingError::QrDecompositionCalculationError;
        assert_eq!(
            error.to_string(),
            "QR Decomposition Calculation Error".to_string()
        );
    }

    #[test]
    fn test_display_model_fit_error() {
        let error = FittingError::ModelFitError;
        assert_eq!(error.to_string(), "Model Fit Error".to_string());
    }
}
