use crate::errors::ModelError;
use crate::structs::RealMatrix;
use crate::traits::{predictor::PredictionResult, Predictor};

/// A linear predictor that multiplies the feature matrix by a parameter matrix.
pub struct LinearPredictor<'a> {
    pub features: &'a RealMatrix,
    pub parameters: RealMatrix,
    pub result: Option<RealMatrix>,
}

impl<'a> LinearPredictor<'a> {
    /// Create a new linear predictor with the given feature matrix and parameter matrix.
    pub fn new(features: &'a RealMatrix, parameters: RealMatrix) -> Self {
        LinearPredictor {
            features,
            parameters,
            result: None,
        }
    }
}

impl<'a> Predictor for LinearPredictor<'a> {
    /// Predict the output values for the given feature matrix and parameter vector.
    fn predict(&mut self) -> PredictionResult {
        self.result = Some(self.features.clone().dot(&mut self.parameters)?);
        self.result.as_ref().ok_or(ModelError::PredictError(
            "Prediction failed for unknown reasons".to_string(),
        ))
    }
}
