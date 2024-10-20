use crate::prelude::ModelData;
use crate::traits::Scorer;

/* #[derive(Debug)]
pub struct MseScorer<'a> {
    data: &'a ModelData,
}

impl<'a> MseScorer<'a> {
    pub fn new(data: &'a ModelData) -> MseScorer {
        MseScorer { data }
    }
}

impl Scorer for MseScorer<'_> {
    /// Calculate the mean squared error between the predictions and the targets.
    fn score(&self, predictions: &RealMatrix) -> f64 {
        let residuals = predictions.sub(&self.data.y());
        let squared_residuals = residuals.apply(|x| x.powi(2));
        let sum_squared_residuals = squared_residuals.sum();
        let n = predictions.rows() as f64;

        sum_squared_residuals / n
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::prelude::RealMatrix;

    fn get_test_x() -> RealMatrix {
        RealMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]])
    }

    fn get_test_y() -> RealMatrix {
        RealMatrix::from_2d_array(&[[1.0], [2.0], [3.0], [4.0]])
    }

    fn get_test_data() -> ModelData {
        let x = get_test_x();
        let y = get_test_y();
        ModelData::new(x, y)
    }

    #[test]
    fn test_mse_scorer() {
        let data = get_test_data();
        let scorer = MseScorer::new(&data);

        let predictions = RealMatrix::from_2d_array(&[[1.0], [2.0], [3.0], [4.0]]);
        let score = scorer.score(&predictions);

        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mse_scorer_when_each_prediction_is_one_unit_too_large() {
        let data = get_test_data();
        let scorer = MseScorer::new(&data);
        let predictions = RealMatrix::from_2d_array(&[[2.0], [3.0], [4.0], [5.0]]);
        let score = scorer.score(&predictions);

        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_mse_scorer_when_each_prediction_is_one_unit_too_small() {
        let data = get_test_data();
        let scorer = MseScorer::new(&data);
        let predictions = RealMatrix::from_2d_array(&[[0.0], [1.0], [2.0], [3.0]]);
        let score = scorer.score(&predictions);

        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_mse_scorer_when_each_prediction_is_two_units_too_large() {
        let data = get_test_data();
        let scorer = MseScorer::new(&data);
        let predictions = RealMatrix::from_2d_array(&[[3.0], [4.0], [5.0], [6.0]]);
        let score = scorer.score(&predictions);

        assert_eq!(score, 4.0);
    }
}
 */
