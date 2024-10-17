pub mod least_squares_qr_decomposition;

use crate::traits::{Fitter, FitterReturn};
use crate::{ModelData, RealMatrix};

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
