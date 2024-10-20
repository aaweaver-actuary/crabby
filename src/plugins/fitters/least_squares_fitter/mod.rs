pub mod least_squares_qr_decomposition;

use crate::prelude::{ModelData, RealMatrix};
use crate::traits::{Fitter, FitterReturn};

use crate::plugins::fitters::least_squares_fitter::least_squares_qr_decomposition::LeastSquaresQrDecompositionFitter;

/// Least squares fitter with initial implementation using QR decomposition,
/// but can be extended to include other methods in the future.
#[derive(Debug)]
pub enum LeastSquaresFitter<'a> {
    QRDecompositionMethod(LeastSquaresQrDecompositionFitter<'a>),
}

impl<'a> LeastSquaresFitter<'a> {
    /// Initialize a default least squares fitter using the QR decomposition method.
    pub fn new(data: &'a ModelData) -> Self {
        LeastSquaresFitter::QRDecompositionMethod(LeastSquaresQrDecompositionFitter::new(data))
    }

    pub fn get_parameters(&self) -> Box<RealMatrix> {
        match self {
            LeastSquaresFitter::QRDecompositionMethod(fitter) => fitter.get_parameters(),
        }
    }
}

impl<'a> Fitter<'a> for LeastSquaresFitter<'a> {
    /// Fit the model to the data using the specified method.
    fn fit(&mut self) -> FitterReturn {
        match self {
            LeastSquaresFitter::QRDecompositionMethod(fitter) => fitter.fit(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::create_real_matrix;

    fn get_testing_modeldata() -> (RealMatrix, RealMatrix) {
        let x = create_real_matrix(vec![1.1, 1.9, 3.05, 3.95, 5.05, 6.1], 3, 2);
        let y = create_real_matrix(vec![1.0, 2.0, 3.0], 3, 1);
        (x, y)
    }

    #[test]
    fn test_least_squares_fitter_new() {
        let (x, y) = get_testing_modeldata();
        let data = ModelData::new(&x, &y);
        let fitter = LeastSquaresFitter::new(&data);
        let is_qr_decomposition = matches!(fitter, LeastSquaresFitter::QRDecompositionMethod(_));

        assert!(is_qr_decomposition);
    }

    #[test]
    fn test_least_squares_fitter_get_parameters() {
        let (x, y) = get_testing_modeldata();
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresFitter::new(&data);
        fitter.fit().unwrap();
        let params = fitter.get_parameters();
        assert_eq!(params.n_rows(), 2);
        assert_eq!(params.n_cols(), 1);
    }

    #[test]
    fn test_least_squares_fitter_fit() {
        let (x, y) = get_testing_modeldata();
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresFitter::new(&data);
        let result = fitter.fit();
        assert!(result.is_ok());
    }
}
