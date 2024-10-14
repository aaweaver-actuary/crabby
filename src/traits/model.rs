use crate::structs::RealMatrix;

pub trait Model {
    /// Fit the model to the training data.
    fn fit(&mut self);

    /// Predict the target variable for the given features.
    fn predict(&self, features: &RealMatrix) -> RealMatrix;

    /// Score the model on the given features.
    fn score(&self, features: &RealMatrix, target: &RealMatrix) -> f64;
}
