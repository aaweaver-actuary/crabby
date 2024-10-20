use crate::prelude::{LossFunction, RealMatrix};

#[derive(Debug)]
pub struct L2LossFunction<'a> {
    y: &'a RealMatrix,
}

impl<'a> L2LossFunction<'a> {
    pub fn new(y: &'a RealMatrix) -> L2LossFunction {
        L2LossFunction { y }
    }
}

impl<'a> LossFunction<'a> for L2LossFunction<'a> {
    /// Compute the gradient of the L2 loss function. The gradient of
    /// the L2 loss function is the difference between the predictions
    /// and the targets.
    fn compute_gradient(&self, predictions: &'a RealMatrix) -> RealMatrix {
        predictions.minus(self.y)
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
    fn compute_loss(&self, predictions: &'a RealMatrix) -> f64 {
        let diff = self.compute_gradient(predictions);
        let squared_diff = diff.iter_rows().fold(0.0, |acc, row| {
            acc + row.iter().fold(0.0, |acc, &val| acc + val * val)
        });
        squared_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_gradient() {
        let y = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let predictions = RealMatrix::from_vec(vec![1.5, 2.5, 3.5], 3, Some(1));
        let loss_function = L2LossFunction::new(&y);

        let gradient = loss_function.compute_gradient(&predictions);
        let expected_gradient = RealMatrix::from_vec(vec![0.5, 0.5, 0.5], 3, Some(1));

        assert_eq!(gradient, expected_gradient);
    }

    #[test]
    fn test_compute_loss() {
        let y = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let predictions = RealMatrix::from_vec(vec![1.5, 2.5, 3.5], 3, Some(1));
        let loss_function = L2LossFunction::new(&y);

        let loss = loss_function.compute_loss(&predictions);
        let expected_loss = 0.75; // (0.5^2 + 0.5^2 + 0.5^2) = 0.75

        assert!((loss - expected_loss).abs() < 1e-6);
    }

    #[test]
    fn test_zero_loss() {
        let y = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let predictions = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let loss_function = L2LossFunction::new(&y);

        let loss = loss_function.compute_loss(&predictions);
        let expected_loss = 0.0;

        assert!((loss - expected_loss).abs() < 1e-6);
    }

    #[test]
    fn test_zero_gradient() {
        let y = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let predictions = RealMatrix::from_vec(vec![1.0, 2.0, 3.0], 3, Some(1));
        let loss_function = L2LossFunction::new(&y);

        let gradient = loss_function.compute_gradient(&predictions);
        let expected_gradient = RealMatrix::from_vec(vec![0.0, 0.0, 0.0], 3, Some(1));

        assert_eq!(gradient, expected_gradient);
    }

    #[test]
    fn test_negative_values() {
        let y = RealMatrix::from_vec(vec![-1.0, -2.0, -3.0], 3, Some(1));
        let predictions = RealMatrix::from_vec(vec![-1.5, -2.5, -3.5], 3, Some(1));
        let loss_function = L2LossFunction::new(&y);

        let gradient = loss_function.compute_gradient(&predictions);
        let expected_gradient = RealMatrix::from_vec(vec![-0.5, -0.5, -0.5], 3, Some(1));

        assert_eq!(gradient, expected_gradient);

        let loss = loss_function.compute_loss(&predictions);
        let expected_loss = 0.75; // (0.5^2 + 0.5^2 + 0.5^2) = 0.75

        assert!((loss - expected_loss).abs() < 1e-6);
    }
}
