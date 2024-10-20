use crate::prelude::{solve_least_squares, ModelData, RealMatrix};
use crate::traits::{Fitter, FitterReturn};

#[derive(Debug)]
pub struct LeastSquaresQrDecompositionFitter<'a> {
    data: &'a ModelData<'a>,
    parameters: Option<Box<RealMatrix>>,
}

impl<'a> LeastSquaresQrDecompositionFitter<'a> {
    pub fn new(data: &'a ModelData) -> Self {
        LeastSquaresQrDecompositionFitter {
            data,
            parameters: None,
        }
    }

    pub fn get_parameters(&self) -> Box<RealMatrix> {
        self.parameters.clone().unwrap()
    }

    pub fn set_parameters(&mut self, result: RealMatrix) {
        let params = Box::new(result);
        self.parameters = Some(params);
    }
}

impl<'a> Fitter<'a> for LeastSquaresQrDecompositionFitter<'a> {
    fn fit(&mut self) -> FitterReturn {
        let x = self.data.x;
        let y = self.data.y;

        let result_coefficients = solve_least_squares(x, y)?.coefficients;

        self.set_parameters(result_coefficients);

        Ok(())
    }
}

#[cfg(test)]

mod tests {

    use super::*;
    use crate::prelude::create_real_matrix;

    #[test]
    fn test_least_squares_qr_decomposition_fitter() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let x = create_real_matrix(x_values, 6, 1);
        let y = create_real_matrix(y_values, 6, 1);
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresQrDecompositionFitter::new(&data);

        fitter.fit().unwrap();

        let expected = create_real_matrix(vec![2.0], 1, 1);
        assert_eq!(*fitter.get_parameters(), expected);
    }

    #[test]
    fn test_least_squares_qr_decomposition_get_parameters() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let x = create_real_matrix(x_values, 6, 1);
        let y = create_real_matrix(y_values, 6, 1);
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresQrDecompositionFitter::new(&data);

        fitter.fit().unwrap();
    }

    /*     #[test]
    fn test_least_squares_qr_decomposition_fitter_set_parameters() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = create_real_matrix(x_values, 2, 3);
        let y = create_real_matrix(y_values, 2, 3);
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresQrDecompositionFitter::new(&data);

        let expected = create_real_matrix(vec![1.0, 1.0, 1.0], 3, 1);
        fitter.set_parameters(expected.clone());

        assert_eq!(*fitter.get_parameters(), expected);
    }

    #[test]
    fn test_least_squares_qr_decomposition_fitter_get_parameters() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = create_real_matrix(x_values, 2, 3);
        let y = create_real_matrix(y_values, 2, 3);
        let data = ModelData::new(&x, &y);
        let mut fitter = LeastSquaresQrDecompositionFitter::new(&data);

        fitter.fit().unwrap();

        let expected = create_real_matrix(vec![1.0, 1.0, 1.0], 3, 1);
        assert_eq!(*fitter.get_parameters(), expected);
    } */
}
