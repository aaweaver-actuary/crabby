pub mod least_squares_qr_decomposition;

use crate::structs::ModelData;
use crate::traits::{Fitter, FitterReturn};

use crate::plugins::fitters::least_squares_fitter::least_squares_qr_decomposition::LeastSquaresQrDecomposition;

/// Least squares fitter with initial implementation using QR decomposition,
/// but can be extended to include other methods in the future.
#[derive(Debug)]
pub enum LeastSquaresFitter {
    QRDecompositionMethod(LeastSquaresQrDecomposition),
}

impl LeastSquaresFitter {
    /// Initialize a default least squares fitter using the QR decomposition method.
    pub fn new() -> Self {
        LeastSquaresFitter::QRDecompositionMethod(LeastSquaresQrDecomposition::new())
    }
}

impl<'a> Fitter<'a> for LeastSquaresFitter {
    /// Fit the model to the data using the specified method.
    fn fit(&self, data: &'a ModelData) -> FitterReturn<'a> {
        match self {
            LeastSquaresFitter::QRDecompositionMethod(fitter) => fitter.fit(data),
        }
    }
}
