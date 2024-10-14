use crate::structs::RealMatrix;

pub trait Scorer {
    fn scorer(&self, predictions: &RealMatrix, targets: Option<&RealMatrix>) -> f64;
}
