use crate::errors::FittingError;
use crate::plugins::predictors::linear_predictor::LinearPredictor;
use crate::structs::{ModelData, RealMatrix};
use crate::traits::{Fitter, FitterReturn};

type QrDecompositionResult = Result<(RealMatrix, RealMatrix), FittingError>;
type LinearSystemSolution = Result<RealMatrix, FittingError>;

#[derive(Debug)]
pub struct LeastSquaresQrDecomposition;

impl LeastSquaresQrDecomposition {
    pub fn new() -> Self {
        LeastSquaresQrDecomposition
    }

    /// Decompose the matrix using the QR decomposition method.
    fn decompose_matrix_with_qr_decomposition(&self, x: &RealMatrix) -> QrDecompositionResult {
        /*         let (q_result, r_result) = x
        .qr()
        .map_err(|_| FittingError::QrDecompositionCalculationError)?; */

        // Ok((q_result, r_result))
        Ok((x.clone(), x.clone()))
    }

    /// Calculate the parameters of the linear system that has already been decomposed.
    fn calculate_parameters(&self, q: &RealMatrix, r: &mut RealMatrix) -> LinearSystemSolution {
        let parameters = r
            .inv()
            .map_err(|_| FittingError::QrDecompositionCalculationError)?
            .dot(&q.transpose());

        Ok(parameters?)
    }
}

impl Default for LeastSquaresQrDecomposition {
    fn default() -> Self {
        LeastSquaresQrDecomposition::new()
    }
}

impl<'a> Fitter<'a> for LeastSquaresQrDecomposition {
    fn fit(&self, data: &'a ModelData) -> FitterReturn<'a> {
        let (q_result, mut r_result) = self.decompose_matrix_with_qr_decomposition(data.x())?;
        let parameters: RealMatrix = self.calculate_parameters(&q_result, &mut r_result)?;
        let mut_features = Box::new(data.x());

        Ok(Box::new(LinearPredictor::new(&mut_features, parameters)))
    }
}
