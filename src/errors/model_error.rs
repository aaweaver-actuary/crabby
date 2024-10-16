use crate::errors::fitting_error::FittingError;
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
        }
    }
}
