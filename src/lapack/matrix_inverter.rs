use crate::errors::LinearAlgebraError;
use crate::structs::RealMatrix;
use lapack::{dgetrf, dgetri};
use ndarray::Array;

type MatrixInversionResult = Result<RealMatrix, LinearAlgebraError>;

/// Public function to invert a RealMatrix.
/// Returns the inverted matrix as a new RealMatrix.
pub fn invert_matrix(matrix: RealMatrix) -> MatrixInversionResult {
    let mut matrix = matrix;
    let mut inverter = MatrixInverter::new(matrix.as_mut());
    inverter.perform_lu_factorization()?;
    inverter.perform_matrix_inversion()?;
    inverter.validate_dimensions_of_result_matrix()?;
    Ok(matrix)
}

/// Struct to encapsulate the matrix inversion process.
struct MatrixInverter<'a> {
    n: i32,
    matrix: &'a mut RealMatrix,
    lda: i32,
    ipiv: Vec<i32>,
    work: Vec<f64>,
}

impl<'a> MatrixInverter<'a> {
    /// Create a new MatrixInverter instance.
    fn new(matrix: &'a mut RealMatrix) -> Self {
        let n = matrix.n_cols() as i32;
        let lda = 1.max(n);

        let ipiv_size = n;
        let ipiv = Self::allocate_pivot_array(ipiv_size as usize);
        let work = Self::allocate_work_array(n as usize);

        if matrix.n_rows() != n as usize {
            panic!("Matrix must be square to invert");
        }

        MatrixInverter {
            n,
            matrix,
            lda,
            ipiv,
            work,
        }
    }

    /// Perform LU factorization using LAPACK's dgetrf routine. This will overwrite the input matrix,
    /// storing the Lower and Upper triangular matrices in the same matrix. The diagonal of the L matrix
    /// is assumed to be all 1s, and the diagonal of the U matrix is the same as the original matrix.
    fn perform_lu_factorization(&mut self) -> Result<(), LinearAlgebraError> {
        unsafe {
            self.call_dgetrf()?;
        }

        Ok(())
    }

    /// Perform matrix inversion using LAPACK's dgetri routine. This will overwrite the input matrix
    /// with the inverted matrix. The input matrix must have been previously factorized using LU factorization.
    fn perform_matrix_inversion(&mut self) -> Result<(), LinearAlgebraError> {
        unsafe {
            self.call_dgetri()?;
        }
        Ok(())
    }

    /// Call dgetrf from LAPACK to perform LU factorization.
    unsafe fn call_dgetrf(&mut self) -> Result<(), LinearAlgebraError> {
        let mut info = 0;

        let a_slice = self
            .matrix
            .values
            .as_slice_mut()
            .expect("Matrix values are missing");

        unsafe {
            dgetrf(self.n, self.n, a_slice, self.lda, &mut self.ipiv, &mut info);
        }

        if info != 0 {
            return Err(LinearAlgebraError::MatrixInverseError(
                "LU factorization failed".to_string(),
            ));
        }

        // Update the matrix with the LU factorization
        let matrix_shape = [self.n as usize, self.n as usize];
        self.matrix.values = Array::from_shape_vec(matrix_shape, a_slice.to_vec()).unwrap();

        Ok(())
    }

    /// Call dgetri from LAPACK to perform matrix inversion.
    unsafe fn call_dgetri(&mut self) -> Result<(), LinearAlgebraError> {
        let lwork = self.work.len() as i32;
        let mut info = 0;

        let a_slice = self
            .matrix
            .values
            .as_slice_mut()
            .expect("Matrix values are missing");

        unsafe {
            dgetri(
                self.n,
                a_slice,
                self.lda,
                &self.ipiv,
                &mut self.work,
                lwork,
                &mut info,
            );
        }

        if info != 0 {
            return Err(LinearAlgebraError::MatrixInverseError(
                "Matrix inversion failed".to_string(),
            ));
        }
        Ok(())
    }

    /// Check that the dimensions of the resulting matrix are correct.
    fn validate_dimensions_of_result_matrix(&self) -> Result<(), LinearAlgebraError> {
        if self.matrix.n_rows() != self.n as usize || self.matrix.n_cols() != self.n as usize {
            return Err(LinearAlgebraError::MatrixInverseError(
                "Resulting matrix has incorrect dimensions".to_string(),
            ));
        }
        Ok(())
    }

    /// Allocate pivot array for LU factorization.
    fn allocate_pivot_array(n: usize) -> Vec<i32> {
        vec![0; n]
    }

    /// Allocate work array for matrix inversion.
    fn allocate_work_array(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_real_matrix(values: Vec<f64>, rows: usize, cols: Option<usize>) -> RealMatrix {
        RealMatrix::from_vec(values, rows, cols)
    }

    #[test]
    fn test_matrix_inverter_initialization_n() {
        let mut matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let inverter = MatrixInverter::new(matrix.as_mut());
        assert_eq!(inverter.n, 2);
    }

    #[test]
    fn test_matrix_inverter_initialization_lda() {
        let mut matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let inverter = MatrixInverter::new(matrix.as_mut());
        assert_eq!(inverter.lda, 2);
    }

    #[test]
    fn test_matrix_inverter_initialization_ipiv() {
        let mut matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let inverter = MatrixInverter::new(matrix.as_mut());
        assert_eq!(inverter.ipiv, vec![0, 0]);
    }

    #[test]
    fn test_matrix_inverter_initialization_work() {
        let mut matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let inverter = MatrixInverter::new(matrix.as_mut());
        assert_eq!(inverter.work, vec![0.0, 0.0]);
    }

    #[test]
    fn test_matrix_inverter_allocate_pivot_array() {
        let result = MatrixInverter::allocate_pivot_array(7);
        assert_eq!(result, vec![0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_matrix_inverter_allocate_work_array() {
        let result = MatrixInverter::allocate_work_array(5);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_invert_matrix() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));
        let result = invert_matrix(matrix).unwrap();
        let expected = create_real_matrix(vec![-2.0, 1.0, 1.5, -0.5], 2, Some(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_more_complex_3_by_3_matrix() {
        let matrix = create_real_matrix(
            vec![1.2, 3.4, 5.6, 7.8, 9.1, 11.12, 13.14, 15.16, 17.18],
            3,
            Some(3),
        );
        let result = invert_matrix(matrix).unwrap();
        let expected = create_real_matrix(
            vec![-0.64, 1.39, -0.69, 0.64, -2.78, 1.59, -0.07, 1.39, -0.82],
            3,
            Some(3),
        );

        // Loop over the values and make sure they are within 1e-1 of each other
        for i in 0..3 {
            for j in 0..3 {
                assert!((result.values[[i, j]] - expected.values[[i, j]]).abs() < 1e-2);
            }
        }
    }
}
