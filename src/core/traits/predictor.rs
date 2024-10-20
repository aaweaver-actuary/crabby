use crate::prelude::{errors::ModelError, RealMatrix};

pub type PredictionResult<'a> = Result<&'a RealMatrix, ModelError>;
pub trait Predictor {
    /// Predict the target values for the given data
    fn predict(&mut self) -> PredictionResult;
}
