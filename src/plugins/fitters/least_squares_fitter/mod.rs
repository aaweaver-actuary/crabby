use crate::errors::ModelError;
use crate::structs::ModelData;
use crate::traits::{Fitter, Predictor};

pub mod least_squares_qr_decomposition;

pub use least_squares_qr_decomposition::LeastSquaresQrDecomposition;

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

impl Fitter for LeastSquaresFitter {
    /// Fit the model to the data using the specified method.
    fn fit(&self, data: &ModelData) -> Result<Box<dyn Predictor>, ModelError> {
        match self {
            LeastSquaresFitter::QRDecompositionMethod(fitter) => fitter.fit(data),
        }
    }
}
