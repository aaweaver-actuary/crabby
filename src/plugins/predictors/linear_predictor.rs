use crate::errors::ModelError;
use crate::structs::RealMatrix;
use crate::traits::{predictor::PredictionResult, Predictor};

/// A linear predictor that multiplies the feature matrix by a parameter matrix.
pub struct LinearPredictor<'a> {
    pub features: &'a RealMatrix,
    pub parameters: &'a RealMatrix,
    pub result: Option<RealMatrix>,
}

impl<'a> LinearPredictor<'a> {
    /// Create a new linear predictor with the given feature matrix and parameter matrix.
    pub fn new(features: &'a RealMatrix, parameters: &'a RealMatrix) -> Self {
        if features.n_cols() != parameters.n_rows() {
            panic!("The number of columns in the feature matrix must be equal to the number of rows in the parameter matrix.");
        }

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
        self.result = Some(self.features.clone().dot(self.parameters)?);
        self.result.as_ref().ok_or(ModelError::PredictError(
            "Prediction failed for unknown reasons".to_string(),
        ))
    }
}

#[cfg(test)]

mod tests {

    use super::*;
    use crate::prelude::create_real_matrix;

    #[test]
    fn test_predict() {
        let features = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let parameters = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut predictor = LinearPredictor::new(&features, &parameters);

        let result = predictor.predict().unwrap();
        let expected = create_real_matrix(vec![7.0, 10.0, 15.0, 22.0], 2, 2);

        assert_eq!(result.clone(), expected);
    }

    #[test]
    fn test_object_construction_features() {
        let features = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let parameters = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let predictor = LinearPredictor::new(&features, &parameters);

        assert_eq!(predictor.features, &features);
    }

    #[test]
    fn test_object_construction_params() {
        let features = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let parameters = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let predictor = LinearPredictor::new(&features, &parameters);

        let predictor_params = predictor.parameters.clone();
        let expected_params = parameters.clone();

        assert_eq!(predictor_params, expected_params);
    }
}
