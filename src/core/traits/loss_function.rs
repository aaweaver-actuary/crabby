use crate::prelude::RealMatrix;

pub trait LossFunction<'a> {
    fn compute_gradient(&self, predictions: &'a RealMatrix) -> RealMatrix;
    fn compute_loss(&self, predictions: &'a RealMatrix) -> f64;
}
