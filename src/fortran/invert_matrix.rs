/* use crate::errors::LinearAlgebraError;
use crate::structs::RealMatrix;
use lapack::{dgetrf, dgetri};
use ndarray::{arr2, Array2};

pub type MatrixInversionResult = Result<RealMatrix, LinearAlgebraError>;
pub type LuFactorizationResult<'a> = Result<&'a mut Array2<f64>, LinearAlgebraError>;
pub type LuFactorizationFortranResult<'a> = Result<(), LinearAlgebraError>;

/// Invert the matrix by first performing LU factorization.
pub fn invert_matrix(x: &'a RealMatrix) -> MatrixInversionResult {
    let mut values = x.clone();
    let mut ipiv = vec![0; values.n_rows() as usize];
    lu_factorize(&mut values.values, &mut ipiv)?;
    invert_from_lu(&mut values, &mut ipiv)
}

/// Perform LU factorization on the given matrix.
pub fn lu_factorize(
    values: &mut Array2<f64>,
    ipiv: &mut Vec<i32>,
) -> Result<(), LinearAlgebraError> {
    let n = values.nrows() as i32;
    let mut info = 0;

    unsafe {
        dgetrf(n, n, values.as_slice_mut().unwrap(), n, ipiv, &mut info);
    }

    if info != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "LU factorization failed".to_string(),
        ));
    }
    Ok(())
}

/// Invert the matrix from its LU factorization.
pub fn invert_from_lu(values: &'a mut RealMatrix, ipiv: &mut Vec<i32>) -> MatrixInversionResult {
    let n = values.nrows() as i32;
    let mut info = 0;
    let mut work = vec![0.0; (n as usize) * 64];
    let lwork = work.len() as i32;

    f_dgetri(
        n,
        values.as_slice_mut().unwrap(),
        n,
        ipiv,
        &mut work,
        lwork,
        &mut info,
    );

    if info != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "Matrix inversion failed".to_string(),
        ));
    }

    Ok(values)
}

/// Perform LU factorization on the given matrix, using the LAPACK dgetrf function.
pub fn f_dgetrf(
    matrix_order: i32,
    matrix: &mut [f64],
    n_rows: i32,
    pivot_indices: &mut [i32],
    return_code: &mut i32,
) -> LuFactorizationFortranResult {
    unsafe {
        dgetrf(
            matrix_order,
            n_rows,
            matrix,
            n_rows,
            pivot_indices,
            return_code,
        );
    }

    if *return_code != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "LU factorization failed".to_string(),
        ));
    }

    Ok(())
}

/// Perform matrix inversion on the given matrix, using the LAPACK dgetri function.
pub fn f_dgetri(
    matrix_order: i32,
    matrix: &mut [f64],
    n_rows: i32,
    pivot_indices: &mut [i32],
    workspace: &mut [f64],
    workspace_length: i32,
    return_code: &mut i32,
) -> MatrixInversionResult {
    unsafe {
        // The LAPACK dgetri function requires a workspace array to be passed in.
        // The inverted matrix will be stored in the input matrix array.
        dgetri(
            matrix_order,
            matrix,
            n_rows,
            pivot_indices,
            workspace,
            workspace_length,
            return_code,
        );
    }

    if *return_code != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "Matrix inversion failed".to_string(),
        ));
    }

    let inverted_vec = Vec::from(matrix);
    Ok(RealMatrix::from_vec(
        inverted_vec,
        n_rows
            .try_into()
            .expect("Unable to convert n_rows to usize"),
        None,
    ))
}

/* #[cfg(test)]

mod tests {
    use super::*;
    use crate::errors::LinearAlgebraError;
    use crate::structs::RealMatrix;
    use ndarray::Array2;

    /// Check if two matrices are equal within a given tolerance.
    fn assert_matrices_are_equal(
        matrix1: &RealMatrix,
        matrix2: Array2<f64>,
        tolerance: Option<f64>,
    ) {
        let matrix1_values = &matrix1.values;
        let matrix2_values = matrix2.as_slice().unwrap();
        let tolerance = tolerance.unwrap_or(1e-6);
        let is_equal = matrix1_values
            .iter()
            .zip(matrix2_values.iter())
            .all(|(a, b)| (a - b).abs() < tolerance);
        assert!(is_equal);
    }

    /// Check if the result of an operation raises an error.
    fn assert_inversion_result_is_error(result: &MatrixInversionResult, error_message: &str) {
        let is_error = match result {
            Err(LinearAlgebraError::MatrixInverseError(message)) => message == error_message,
            _ => false,
        };

        assert!(is_error);
    }

    fn assert_lu_factorization_result_is_error(
        result: &LuFactorizationFortranResult,
        error_message: &str,
    ) {
        let is_error = match result {
            Err(LinearAlgebraError::MatrixInverseError(message)) => message == error_message,
            _ => false,
        };

        assert!(is_error);
    }

    #[test]
    fn test_invert_matrix() {
        let matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let expected = arr2(&[[-2.0, 1.0], [1.5, -0.5]]);
        let result_of_matrix_inversion = invert_matrix(&RealMatrix::new(matrix.clone())).unwrap();

        assert_matrices_are_equal(&result_of_matrix_inversion, expected, None);
    }

    #[test]
    fn test_invert_matrix_error() {
        let matrix = arr2(&[[1.0, 2.0], [1.0, 2.0]]);
        let result_of_matrix_inversion = invert_matrix(&RealMatrix::new(matrix));

        assert_inversion_result_is_error(&result_of_matrix_inversion, "Matrix inversion failed");
    }

    #[test]
    fn test_lu_factorize() {
        let mut matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut ipiv = vec![0; 2];
        let expected = arr2(&[[3.0, 4.0], [0.3333333333333333, -0.3333333333333333]]);
        lu_factorize(&mut matrix, &mut ipiv).unwrap();

        assert_matrices_are_equal(&RealMatrix::new(matrix), expected, Some(1e-6));
    }

    #[test]
    fn test_lu_factorize_error() {
        let mut matrix = arr2(&[[1.0, 2.0], [1.0, 2.0]]);
        let mut ipiv = vec![0; 2];
        let lu_factorization_result = lu_factorize(&mut matrix, &mut ipiv);

        assert_lu_factorization_result_is_error(
            &lu_factorization_result,
            "LU factorization failed",
        );
    }

    #[test]
    fn test_invert_from_lu() {
        let mut matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut ipiv = vec![0; 2];
        let expected = arr2(&[[-2.0, 1.0], [1.5, -0.5]]);
        lu_factorize(&mut matrix, &mut ipiv).unwrap();
        invert_from_lu(&mut matrix, &mut ipiv).unwrap();

        assert_matrices_are_equal(&RealMatrix::new(matrix), expected, None);
    }

    #[test]
    fn test_invert_from_lu_when_original_matrix_is_singular() {
        let mut singular_matrix = arr2(&[[1.0, 2.0], [1.0, 2.0]]);
        let mut ipiv = vec![0; 2];
        let attempted_inverse_of_singular_matrix = invert_from_lu(&mut singular_matrix, &mut ipiv);

        assert_inversion_result_is_error(
            &attempted_inverse_of_singular_matrix,
            "Matrix inversion failed",
        );
    }

    #[test]
    fn test_f_dgetrf_works_as_expected() {
        let mut matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut ipiv = vec![0; 2];
        let mut info = 0;
        let expected = arr2(&[[3.0, 4.0], [0.3333333333333333, -0.3333333333333333]]);
        f_dgetrf(1, matrix.as_slice_mut().unwrap(), 2, &mut ipiv, &mut info);

        assert_matrices_are_equal(&RealMatrix::new(matrix), expected, Some(1e-6));
    }

    #[test]
    fn test_f_dgetri_works_as_expected() {
        let mut matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut ipiv = vec![0; 2];
        let mut info = 0;
        let mut work = vec![0.0; 128];
        let lwork = work.len() as i32;
        f_dgetri(
            1,
            matrix.as_slice_mut().unwrap(),
            2,
            &mut ipiv,
            &mut work,
            lwork,
            &mut info,
        );

        let expected = arr2(&[[-2.0, 1.0], [1.5, -0.5]]);
        assert_matrices_are_equal(&RealMatrix::new(matrix), expected, None);
    }

    #[test]
    fn test_f_dgetri_when_original_matrix_is_singular() {
        let mut singular_matrix = arr2(&[[1.0, 2.0], [1.0, 2.0]]);
        let mut ipiv = vec![0; 2];
        let mut info = 0;
        let mut work = vec![0.0; 128];
        let lwork = work.len() as i32;
        f_dgetri(
            1,
            singular_matrix.as_slice_mut().unwrap(),
            2,
            &mut ipiv,
            &mut work,
            lwork,
            &mut info,
        );

        assert_inversion_result_is_error(Err(&singular_matrix), "Matrix inversion failed");
    }

    #[test]
    fn test_f_dgetri_when_lwork_is_too_small() {
        let mut matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut ipiv = vec![0; 2];
        let mut info = 0;
        let mut work = vec![0.0; 1];
        let lwork = work.len() as i32;
        let attempted_inverse_of_singular_matrix = f_dgetri(
            1,
            matrix.as_slice_mut().unwrap(),
            2,
            &mut ipiv,
            &mut work,
            lwork,
            &mut info,
        );

        assert_inversion_result_is_error(
            &attempted_inverse_of_singular_matrix,
            "Matrix inversion failed",
        );
    }

    #[test]
    fn test_f_dgetri_when_info_is_not_zero() {
        let mut matrix = arr2(&[[1.0, 2.0], [1.0, 2.0]]);
        let mut ipiv = vec![0; 2];
        let mut info = 1;
        let mut work = vec![0.0; 128];
        let lwork = work.len() as i32;
        let attempted_inverse_of_singular_matrix = f_dgetri(
            1,
            matrix.as_slice_mut().unwrap(),
            2,
            &mut ipiv,
            &mut work,
            lwork,
            &mut info,
        );

        assert_inversion_result_is_error(
            &attempted_inverse_of_singular_matrix,
            "Matrix inversion failed",
        );
    }
}
 */
 */
