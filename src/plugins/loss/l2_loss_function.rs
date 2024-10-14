use crate::structs::RealMatrix;

#[derive(Debug)]
pub struct L2LossFunction<'a> {
    y: &'a RealMatrix,
}

impl L2LossFunction {
    pub fn new(y: &'a RealMatrix) -> L2LossFunction {
        L2LossFunction { y }
    }
}

impl LossFunction for L2LossFunction {
    /// Compute the gradient of the L2 loss function. The gradient of
    /// the L2 loss function is the difference between the predictions
    /// and the targets.
    fn gradient(&self, y_hat: &'a RealMatrix) -> RealMatrix {
        y_hat.sub(&self.y)
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
    fn loss(&self, y_hat: &'a RealMatrix) -> RealMatrix {
        let diff = self.gradient(y_hat);
        diff.mul_element_wise(&diff).sum_rows().sqrt()
    }
}
