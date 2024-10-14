use crate::errors::ModelError;
use crate::structs::RealMatrix;

pub trait Predictor {
    /// Predict the target values for the given data
    fn predict(&mut self) -> Result<&RealMatrix, ModelError>;
}
