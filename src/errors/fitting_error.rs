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
        }
    }
}
