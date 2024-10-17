use crate::errors::LinearAlgebraError;
use crate::lapack::blas_transpose_flag::BlasTransposeFlag;
use crate::structs::{create_real_matrix, RealMatrix};
use blas::dgemm;

pub type MatrixMultiplicationResult = Result<RealMatrix, LinearAlgebraError>;
type InPlaceResult = Result<(), LinearAlgebraError>;

/// Public function to multiply two RealMatrix instances.
/// Returns the result as a new RealMatrix.
pub fn multiply_matrices<'a>(a: &'a RealMatrix, b: &'a RealMatrix) -> MatrixMultiplicationResult {
    let mut multiplier = MatrixMultiplier::new(a, b);
    multiplier.validate_dimensions_of_input_matrices_allow_multiplication()?;
    multiplier.run_blas_dgemm_to_multiply_matrices_in_place()?;

    Ok(multiplier.result_matrix_c)
}

/// A struct to encapsulate the matrix multiplication logic and simplify the large number of variables.
pub struct MatrixMultiplier<'a> {
    pub input_matrix_a: &'a RealMatrix,
    pub input_matrix_b: &'a RealMatrix,
    pub result_matrix_c: RealMatrix,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    alpha: f64,
    beta: f64,
    transpose_a_flag: BlasTransposeFlag,
    transpose_b_flag: BlasTransposeFlag,
}

impl<'a> MatrixMultiplier<'a> {
    pub fn new(a: &'a RealMatrix, b: &'a RealMatrix) -> Self {
        let a_rows = a.n_rows();
        let a_cols = a.n_cols();
        let b_rows = b.n_rows();
        let b_cols = b.n_cols();

        MatrixMultiplier {
            input_matrix_a: a,
            input_matrix_b: b,
            result_matrix_c: RealMatrix::with_shape(a_rows, b_cols),
            m: a_rows as i32,
            n: b_cols as i32,
            k: a_cols as i32,
            lda: a_rows as i32,
            ldb: b_rows as i32,
            ldc: a_rows as i32,
            alpha: 1.0,
            beta: 0.0,
            transpose_a_flag: BlasTransposeFlag::NoTranspose,
            transpose_b_flag: BlasTransposeFlag::NoTranspose,
        }
    }

    fn validate_dimensions_of_input_matrices_allow_multiplication(&self) -> InPlaceResult {
        if self.input_matrix_a.n_cols() != self.input_matrix_b.n_rows() {
            return Err(LinearAlgebraError::DimensionMismatchError(
                "Matrix dimensions are not compatible for multiplication".to_string(),
            ));
        }
        Ok(())
    }

    pub fn run_blas_dgemm_to_multiply_matrices_in_place(&mut self) -> InPlaceResult {
        unsafe {
            self.call_blas_dgemm_to_multiply_matrices().map_err(|_| {
                LinearAlgebraError::OperationFailedError("dgemm call failed".to_string())
            })
        }
    }

    /// Call BLAS's dgemm function to perform matrix multiplication
    unsafe fn call_blas_dgemm_to_multiply_matrices(&mut self) -> Result<(), ()> {
        let a_slice = self.input_matrix_a.values.as_slice().unwrap();
        let b_slice = self.input_matrix_b.values.as_slice().unwrap();
        let c_slice = self.result_matrix_c.values.as_slice_mut().unwrap();

        dgemm(
            self.transpose_a_flag.to_blas_char(),
            self.transpose_b_flag.to_blas_char(),
            self.m,
            self.n,
            self.k,
            self.alpha,
            a_slice,
            self.lda,
            b_slice,
            self.ldb,
            self.beta,
            c_slice,
            self.ldc,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_matrix_a() -> RealMatrix {
        create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
    }

    fn get_matrix_b() -> RealMatrix {
        create_real_matrix(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3)
    }

    fn get_expected_result() -> RealMatrix {
        create_real_matrix(vec![21.0, 24.0, 27.0, 47.0, 54.0, 61.0], 2, 3)
    }

    #[test]
    fn test_matrix_multiplier_is_constructed_correctly() {
        let matrix_a = get_matrix_a();
        let matrix_b = get_matrix_b();
        let multiplier = MatrixMultiplier::new(&matrix_a, &matrix_b);

        assert_eq!(multiplier.m, 2);
        assert_eq!(multiplier.n, 3);
        assert_eq!(multiplier.k, 2);
        assert_eq!(multiplier.lda, 2);
        assert_eq!(multiplier.ldb, 2);
        assert_eq!(multiplier.ldc, 2);
        assert_eq!(multiplier.alpha, 1.0);
        assert_eq!(multiplier.beta, 0.0);
        assert_eq!(multiplier.transpose_a_flag, BlasTransposeFlag::NoTranspose);
        assert_eq!(multiplier.transpose_b_flag, BlasTransposeFlag::NoTranspose);
    }

    #[test]
    fn test_that_matrix_multiplier_does_not_permit_incorrect_dimensions() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2); // 2x2 matrix
        let matrix_b = create_real_matrix(vec![5.0, 6.0, 7.0, 8.0], 1, 4); // 1x4 matrix
        let multiplier = MatrixMultiplier::new(&matrix_a, &matrix_b); // 2x2 * 1x4 should fail

        let result = multiplier.validate_dimensions_of_input_matrices_allow_multiplication();
        assert!(result.is_err());
    }

    #[test]
    fn test_that_result_matrix_is_allocated_correctly() {
        let matrix_a = get_matrix_a();
        let matrix_b = get_matrix_b();
        let multiplier = MatrixMultiplier::new(&matrix_a, &matrix_b);
        let result = multiplier.result_matrix_c;

        assert_eq!(result.n_rows(), 2);
        assert_eq!(result.n_cols(), 3);

        for row in 0..result.n_rows() {
            for col in 0..result.n_cols() {
                assert_eq!(result.values[[row, col]], 0.0);
            }
        }
    }

    #[test]
    fn test_that_result_matrix_is_the_correct_size() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2); // 2x2 matrix
        let matrix_b = create_real_matrix(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3); // 2x3 matrix
        let multiplier = MatrixMultiplier::new(&matrix_a, &matrix_b);

        let result = multiplier.result_matrix_c;
        assert_eq!(result.n_rows(), 2);
        assert_eq!(result.n_cols(), 3);
    }

    #[test]
    fn test_that_perform_dgemm_correctly_modifies_result_matrix() {
        let matrix_a = get_matrix_a();
        let matrix_b = get_matrix_b();

        let mut multiplier = MatrixMultiplier::new(&matrix_a, &matrix_b);
        multiplier
            .run_blas_dgemm_to_multiply_matrices_in_place()
            .unwrap();

        let expected_result = get_expected_result();
        let result = multiplier.result_matrix_c;
        println!("\n\nactual: {:?}", result);
        println!("\n\nexpected: {:?}", expected_result);

        assert_eq!(result.values, expected_result.values);
    }

    #[test]
    fn test_public_multiply_matrices_function() {
        let matrix_a = get_matrix_a();
        let matrix_b = get_matrix_b();

        let result = multiply_matrices(&matrix_a, &matrix_b).unwrap();
        let expected_result = get_expected_result();

        assert_eq!(result.values, expected_result.values);
    }
}
