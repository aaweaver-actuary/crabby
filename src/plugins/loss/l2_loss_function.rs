use crate::structs::RealMatrix;
use crate::traits::LossFunction;

#[derive(Debug)]
pub struct L2LossFunction<'a> {
    y: &'a RealMatrix,
}

impl<'a> L2LossFunction<'a> {
    pub fn new(y: &'a RealMatrix) -> L2LossFunction {
        L2LossFunction { y }
    }
}

/* impl<'a> LossFunction<'a> for L2LossFunction<'a> {
    /// Compute the gradient of the L2 loss function. The gradient of
    /// the L2 loss function is the difference between the predictions
    /// and the targets.
    fn compute_gradient(&self, predictions: &RealMatrix, targets: &RealMatrix) -> RealMatrix {

        predictions.minus(targets)
    }

    /// Compute the L2 loss between the predictions and the targets. The L2 loss
    /// is defined as the square root of the sum of the squared differences between
    /// the predictions and the targets (eg the Mean Squared Error).
    ///
    /// # Arguments
    /// * `y` - A matrix of shape (n_samples, 1) containing the target values.
    /// * `y_hat` - A matrix of shape (n_samples, 1) containing the predicted values.
    ///
    /// # Returns
    /// The L2 loss between the predictions and the targets.
    fn compute_loss(&self, predictions: &'a RealMatrix, targets: &'a RealMatrix) -> f64 {
        let diff = self.compute_gradient(&predictions   , &targets);
        let squared_diff = diff.into_iter().map(|x| x.powi(2));
    }
}
 */
