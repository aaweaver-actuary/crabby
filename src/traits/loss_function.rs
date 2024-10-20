use crate::structs::RealMatrix;

pub trait LossFunction {
    fn compute_gradient(&self, predictions: &RealMatrix, targets: &RealMatrix) -> RealMatrix;
    fn compute_loss(&self, predictions: &RealMatrix, targets: &RealMatrix) -> f64;
}
