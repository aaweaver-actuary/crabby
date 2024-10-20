use crate::errors::LinearAlgebraError;
use crate::structs::RealMatrix;
use lapack::dgels;
use std::cmp::max;

use super::BlasTransposeFlag;

type LeastSquaresResult = Result<LeastSquaresSolution, LinearAlgebraError>;

/// Public function to solve the least squares problem.
/// Returns the fitted coefficients and other statistics as a LeastSquaresSolution.
pub fn solve_least_squares(x: &RealMatrix, y: &RealMatrix) -> LeastSquaresResult {
    let mut solver = LeastSquaresSolver::new(x, y)?;
    solver.perform_least_squares()?;
    solver.extract_solution()
}

/// Struct to encapsulate the least squares solving process.
struct LeastSquaresSolver {
    m: i32,
    n: i32,
    nrhs: i32,
    a: Vec<f64>,
    lda: i32,
    b: Vec<f64>,
    ldb: i32,
    work: Vec<f64>,
    lwork: i32,
}

impl LeastSquaresSolver {
    /// Create a new LeastSquaresSolver instance.
    fn new(x: &RealMatrix, y: &RealMatrix) -> Result<Self, LinearAlgebraError> {
        let m = x.n_rows() as i32;
        let n = x.n_cols() as i32;
        let nrhs = y.n_cols() as i32;

        let lda = max(1, m);
        let ldb = max(1, max(m, n));

        let a = x.values.clone().into_raw_vec_and_offset().0;
        let b = y.values.clone().into_raw_vec_and_offset().0;

        let mut solver = LeastSquaresSolver {
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            work: Vec::new(),
            lwork: -1, // Will be set after querying optimal workspace size
        };

        solver.validate_input_dimensions(y)?;
        solver.query_optimal_workspace_size()?;

        Ok(solver)
    }

    /// Validate that the input dimensions are compatible.
    fn validate_input_dimensions(&self, y: &RealMatrix) -> Result<(), LinearAlgebraError> {
        if y.n_rows() as i32 != self.m {
            return Err(LinearAlgebraError::DimensionMismatchError(
                "Number of rows in x and y must match".to_string(),
            ));
        }
        Ok(())
    }

    /// Query the optimal workspace size and allocate the work array.
    fn query_optimal_workspace_size(&mut self) -> Result<(), LinearAlgebraError> {
        let mut work = vec![0.0];
        let mut info = 0;
        let mut a_clone = self.a.clone();
        let mut b_clone = self.b.clone();
        unsafe {
            dgels(
                BlasTransposeFlag::NoTranspose.to_blas_char(),
                self.m,
                self.n,
                self.nrhs,
                &mut a_clone,
                self.lda,
                &mut b_clone,
                self.ldb,
                &mut work,
                -1,
                &mut info,
                1,
            );
        }
        if info != 0 {
            return Err(LinearAlgebraError::LapackError(
                "dgels workspace query failed".to_string(),
            ));
        }
        self.lwork = work[0] as i32;
        self.work = vec![0.0; self.lwork as usize];
        Ok(())
    }

    /// Perform least squares solving using LAPACK's dgels routine.
    fn perform_least_squares(&mut self) -> Result<(), LinearAlgebraError> {
        let mut info = 0;
        unsafe {
            dgels(
                BlasTransposeFlag::NoTranspose.to_blas_char(),
                self.m,
                self.n,
                self.nrhs,
                &mut self.a,
                self.lda,
                &mut self.b,
                self.ldb,
                &mut self.work,
                self.lwork,
                &mut info,
                1,
            );
        }
        if info != 0 {
            return Err(LinearAlgebraError::LapackError(format!(
                "DGELS failed with info = {}",
                info
            )));
        }
        Ok(())
    }

    /// Extract the solution from the solver and return as a LeastSquaresSolution.
    fn extract_solution(&self) -> Result<LeastSquaresSolution, LinearAlgebraError> {
        let n = self.n as usize;
        let nrhs = self.nrhs as usize;

        // Extract coefficients from b
        let coefficients = RealMatrix::from_vec(self.b[0..n * nrhs].to_vec(), n, Some(nrhs));

        Ok(LeastSquaresSolution {
            coefficients,
            // Standard errors can be added here if needed
        })
    }
}

/// Struct to hold the results of the least squares solution.
pub struct LeastSquaresSolution {
    pub coefficients: RealMatrix,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_real_matrix(values: Vec<f64>, rows: usize, cols: Option<usize>) -> RealMatrix {
        RealMatrix::from_vec(values, rows, cols)
    }

    #[test]
    fn test_least_squares_solver_initialization() {
        let x = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let y = create_real_matrix(vec![5.0, 6.0], 2, Some(1));
        let solver = LeastSquaresSolver::new(&x, &y).unwrap();
        assert_eq!(solver.m, 2);
        assert_eq!(solver.n, 2);
        assert_eq!(solver.nrhs, 1);
    }

    #[test]
    fn test_solve_least_squares() {
        let x = create_real_matrix(vec![1.0, 1.0, 1.0, 2.0], 2, Some(2));
        let y = create_real_matrix(vec![1.0, 2.0], 2, Some(1));
        let result = solve_least_squares(&x, &y).unwrap();
        let expected_coefficients = create_real_matrix(vec![0.0, 1.0], 2, Some(1));
        assert_eq!(result.coefficients, expected_coefficients);
    }

    #[test]
    fn test_solve_least_squares_overdetermined() {
        let x = create_real_matrix(
            vec![
                0.349, 0.489, -0.442, -0.776, 0.041, 0.672, 0.311, -0.088, -0.543, -0.77, 1.647,
                0.498,
            ],
            6,
            Some(2),
        );
        let y = create_real_matrix(
            vec![1.1195, -1.7725, -0.906, -1.642, 4.5575, 0.933],
            6,
            Some(1),
        );
        let result = solve_least_squares(&x, &y).unwrap();
        let coefficients = &result.coefficients.values.as_slice().unwrap();
        println!("coefficients: {:?}", coefficients);

        let coef_1_diff = (coefficients[0] + 1.158).abs();
        let coef_2_diff = (coefficients[1] - 2.992).abs();

        println!("coef_1_diff: {}", coef_1_diff);
        println!("coef_2_diff: {}", coef_2_diff);
        assert!((coefficients[0] + 1.157).abs() < 1e-2);
        assert!((coefficients[1] - 2.97).abs() < 1e-2);
    }

    #[test]
    fn test_validate_input_dimensions_error_case() {
        let x = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let y = create_real_matrix(vec![5.0, 6.0, 7.0], 3, Some(1));
        let solver = LeastSquaresSolver::new(&x, &y);

        assert!(solver.is_err());
    }
}
