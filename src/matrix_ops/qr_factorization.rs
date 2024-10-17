use crate::errors::LinearAlgebraError;
use crate::structs::RealMatrix;
use lapack::{dgeqrt3, dorgqr};

type QrFactorizationResult = Result<(RealMatrix, RealMatrix), LinearAlgebraError>;
type DgeqrtResult = Result<Vec<f64>, LinearAlgebraError>;
type Dgeqrt3Result = Result<(), LinearAlgebraError>;
type DorgqrResult = Result<(), LinearAlgebraError>;

/// A struct to represent the dimensions of a matrix.
struct MatrixDims(i32, i32);

impl MatrixDims {
    /// Get the values of the matrix dimensions.
    fn vals(&self) -> (i32, i32) {
        (self.0, self.1)
    }
}

/// Public function to perform QR factorization on a RealMatrix.
/// Returns a tuple with the Q and R matrices.
pub fn qr_factorization(x: &RealMatrix) -> QrFactorizationResult {
    // Step 1: Convert to column-major format
    let mut column_major_matrix = RealMatrix::new(x.values.clone()).to_column_major();

    // Step 2: Perform QR factorization using LAPACK
    let t_matrix = factorize_with_dgeqrt3(&mut column_major_matrix)?;

    // Step 3: Construct Q and R matrices from the results
    generate_q_matrix_in_place(&mut column_major_matrix, &t_matrix)?;
    let r_matrix = extract_r_matrix_in_place(&mut column_major_matrix);

    Ok((column_major_matrix, r_matrix))
}

/// Perform the QR factorization using LAPACK's dgeqrt3 routine.
/// Modifies the given matrix in place and returns the T matrix.
fn factorize_with_dgeqrt3(matrix: &mut RealMatrix) -> DgeqrtResult {
    let (m, n) = get_matrix_dimensions(matrix).vals();
    let mut t = allocate_t_matrix(n);
    let mut info = 0;

    perform_dgeqrt3(matrix, m, n, &mut t, &mut info)?;

    Ok(t) // Return the T matrix
}

/// Get the dimensions of the matrix.
fn get_matrix_dimensions(matrix: &RealMatrix) -> MatrixDims {
    MatrixDims(matrix.n_rows() as i32, matrix.n_cols() as i32)
}

/// Allocate memory for the T matrix based on the number of columns.
fn allocate_t_matrix(n: i32) -> Vec<f64> {
    vec![0.0; n as usize]
}

/// Perform the dgeqrt3 operation using LAPACK.
fn perform_dgeqrt3(
    matrix: &mut RealMatrix,
    m: i32,
    n: i32,
    t: &mut [f64],
    info: &mut i32,
) -> Dgeqrt3Result {
    println!("matrix: {:#?}", matrix);
    println!("m: {}, n: {}", m, n);
    println!("t: {:?}", t);

    let a1 = matrix.as_slice();
    println!("a1: {:?}", a1);

    let a = matrix.as_slice_mut();

    println!("a: {:?}", a);

    let lda = m;
    let ldt = n;

    unsafe {
        dgeqrt3(m, n, a.unwrap(), lda, t, ldt, info);
    }

    if *info != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "QR factorization failed".to_string(),
        ));
    }

    Ok(())
}

/// Generate the Q matrix from the packed data produced by LAPACK.
/// Uses LAPACK's dorgqr routine to generate the orthonormal Q matrix in place.
fn generate_q_matrix_in_place(factored_matrix: &mut RealMatrix, t: &[f64]) -> DorgqrResult {
    let (m, n) = get_matrix_dimensions(factored_matrix).vals();
    let k = n; // The number of elementary reflectors
    let mut info = 0;

    perform_dorgqr(factored_matrix, m, n, k, t, &mut info)?;

    Ok(())
}

/// Allocate memory for the work array required by dorgqr.
fn allocate_work_array(n: i32) -> Vec<f64> {
    vec![0.0; n as usize]
}

/// Perform the dorgqr operation using LAPACK.
fn perform_dorgqr(
    factored_matrix: &mut RealMatrix,
    m: i32,
    n: i32,
    k: i32,
    t: &[f64],
    info: &mut i32,
) -> DorgqrResult {
    let lda = m;
    let a = &mut factored_matrix.as_slice_mut().unwrap();
    let work = &mut allocate_work_array(n);
    let lwork = n;
    let tau = t;

    unsafe {
        /* pub unsafe fn dorgqr(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [f64],
        lda: i32,
        tau: &[f64],
        work: &mut [f64],
        lwork: i32,
        info: &mut i32, */
        dorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    if *info != 0 {
        return Err(LinearAlgebraError::MatrixInverseError(
            "Failed to generate Q matrix".to_string(),
        ));
    }

    Ok(())
}

/// Extract the R matrix from the factored matrix in place.
/// The R matrix is represented by the upper triangular part of the factored matrix.
fn extract_r_matrix_in_place(factored_matrix: &mut RealMatrix) -> RealMatrix {
    let (rows, cols) = get_matrix_dimensions(factored_matrix).vals();
    let mut r_values = allocate_r_matrix(cols);

    fill_r_matrix(
        &factored_matrix
            .as_slice_mut()
            .expect("Failed to convert to a mutable slice"),
        cols,
        &mut r_values,
    );

    RealMatrix::from_slice(&r_values, rows as usize, Some(cols as usize))
}

/// Allocate memory for the R matrix, considering only the upper triangular part.
fn allocate_r_matrix(cols: i32) -> Vec<f64> {
    Vec::with_capacity(((cols * (cols + 1)) / 2).try_into().unwrap())
}

/// Fill the R matrix values from the factored matrix.
fn fill_r_matrix(factored_values: &[f64], cols: i32, r_values: &mut Vec<f64>) {
    factored_values
        .chunks_exact(cols as usize)
        .enumerate()
        .for_each(|(i, row)| {
            r_values.extend(row[i..].iter().cloned());
        });
}

#[cfg(test)]
mod tests {

    use super::*;

    // The following set of A = QR tests are based on the example from
    // https://en.wikipedia.org/wiki/QR_decomposition#Example

    fn sample_a() -> RealMatrix {
        RealMatrix::from_vec(
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            3,
            Some(3),
        )
    }

    fn result_q() -> RealMatrix {
        RealMatrix::from_vec(
            vec![
                6.0 / 7.0,
                -69.0 / 175.0,
                58.0 / 175.0,
                3.0 / 7.0,
                158.0 / 175.0,
                6.0 / 175.0,
                -2.0 / 7.0,
                6.0 / 35.0,
                -33.0 / 35.0,
            ],
            3,
            Some(3),
        )
    }

    fn result_r() -> RealMatrix {
        RealMatrix::from_vec(
            vec![14.0, 21.0, -14.0, 0.0, 175.0, -70.0, 0.0, 0.0, 35.0],
            3,
            Some(3),
        )
    }

    #[test]
    fn test_qr_factorization() {
        let mut a = sample_a();
        let (q, r) = qr_factorization(&mut a).unwrap();

        for q_row in 0..q.n_rows() {
            for q_col in 0..q.n_cols() {
                println!("Q, row: {}, col: {}", q_row, q_col);
                println!("calculated q: {}", q.values[[q_row, q_col]]);
                println!("expected q: {}", result_q().values[[q_row, q_col]]);
                assert!(
                    (q.values[[q_row, q_col]] - result_q().values[[q_row, q_col]]).abs() < 1e-6
                );
            }
        }

        for r_row in 0..r.n_rows() {
            for r_col in 0..r.n_cols() {
                println!("R, row: {}, col: {}", r_row, r_col);
                println!("calculated r: {}", r.values[[r_row, r_col]]);
                println!("expected r: {}", result_r().values[[r_row, r_col]]);
                assert!(
                    (r.values[[r_row, r_col]] - result_r().values[[r_row, r_col]]).abs() < 1e-6
                );
            }
        }

        /*         assert_eq!(q, result_q());
        assert_eq!(r, result_r()); */
    }
}
