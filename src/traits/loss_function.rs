use crate::structs::RealMatrix;

pub trait LossFunction {
    fn compute_loss(&self, predictions: &RealMatrix, targets: &RealMatrix) -> f64;
}
