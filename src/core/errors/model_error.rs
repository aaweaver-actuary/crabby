use crate::prelude::errors::FittingError;
use std::fmt;

use super::LinearAlgebraError;

#[derive(Debug)]
pub enum ModelError {
    DataError(String),
    FitError(String),
    PredictError(String),
    EvaluationError(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ModelError::DataError(ref err) => write!(f, "Data Error: {}", err),
            ModelError::FitError(ref err) => write!(f, "Fit Error: {}", err),
            ModelError::PredictError(ref err) => write!(f, "Predict Error: {}", err),
            ModelError::EvaluationError(ref err) => write!(f, "Evaluation Error: {}", err),
        }
    }
}

impl From<FittingError> for ModelError {
    fn from(error: FittingError) -> Self {
        match error {
            FittingError::QrDecompositionCalculationError => {
                ModelError::FitError("QR Decomposition Calculation Error".to_string())
            }
            FittingError::ModelFitError => ModelError::FitError("Model Fit Error".to_string()),
        }
    }
}

impl From<LinearAlgebraError> for ModelError {
    fn from(error: LinearAlgebraError) -> Self {
        match error {
            LinearAlgebraError::QrDecompositionError(_err) => {
                ModelError::FitError("QR Decomposition Error".to_string())
            }
            LinearAlgebraError::MatrixInverseError(_err) => {
                ModelError::FitError("Matrix Inverse Error".to_string())
            }
            LinearAlgebraError::DotProductError(_err) => {
                ModelError::FitError("Dot Product Error".to_string())
            }
            LinearAlgebraError::DimensionMismatchError(_err) => {
                ModelError::FitError("Dimension Mismatch Error".to_string())
            }
            LinearAlgebraError::OperationFailedError(_err) => {
                ModelError::FitError("Operation Failed Error".to_string())
            }
            LinearAlgebraError::LapackError(_err) => {
                ModelError::FitError("LAPACK Error".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_error_display() {
        let data_error = ModelError::DataError("Invalid data".to_string());
        assert_eq!(format!("{}", data_error), "Data Error: Invalid data");

        let fit_error = ModelError::FitError("Fit failed".to_string());
        assert_eq!(format!("{}", fit_error), "Fit Error: Fit failed");

        let predict_error = ModelError::PredictError("Prediction failed".to_string());
        assert_eq!(
            format!("{}", predict_error),
            "Predict Error: Prediction failed"
        );

        let evaluation_error = ModelError::EvaluationError("Evaluation failed".to_string());
        assert_eq!(
            format!("{}", evaluation_error),
            "Evaluation Error: Evaluation failed"
        );
    }

    #[test]
    fn test_model_error_from_fitting_error() {
        let fitting_error = FittingError::QrDecompositionCalculationError;
        let model_error: ModelError = fitting_error.into();
        assert_eq!(
            format!("{}", model_error),
            "Fit Error: QR Decomposition Calculation Error"
        );

        let fitting_error = FittingError::ModelFitError;
        let model_error: ModelError = fitting_error.into();
        assert_eq!(format!("{}", model_error), "Fit Error: Model Fit Error");
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_qr_deomposition() {
        let la_error = LinearAlgebraError::QrDecompositionError("QR error".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(
            format!("{}", model_error),
            "Fit Error: QR Decomposition Error"
        );
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_matrix_inversion() {
        let la_error = LinearAlgebraError::MatrixInverseError("Inverse error".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(
            format!("{}", model_error),
            "Fit Error: Matrix Inverse Error"
        );
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_dot_product() {
        let la_error = LinearAlgebraError::DotProductError("Dot product error".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(format!("{}", model_error), "Fit Error: Dot Product Error");
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_dimension_mismatch() {
        let la_error = LinearAlgebraError::DimensionMismatchError("Dimension mismatch".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(
            format!("{}", model_error),
            "Fit Error: Dimension Mismatch Error"
        );
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_operation_failed() {
        let la_error = LinearAlgebraError::OperationFailedError("Operation failed".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(
            format!("{}", model_error),
            "Fit Error: Operation Failed Error"
        );
    }

    #[test]
    fn test_model_error_from_linear_algebra_error_via_lapack_error() {
        let la_error = LinearAlgebraError::LapackError("LAPACK error".to_string());
        let model_error: ModelError = la_error.into();
        assert_eq!(format!("{}", model_error), "Fit Error: LAPACK Error");
    }
}
