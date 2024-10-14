// src/real_matrix.rs

use crate::errors::LinearAlgebraError;
use lapack::{dgeqrf, dgetrf, dgetri, dorgqr};
use ndarray::Array2;

type MatrixInversionResult = Result<RealMatrix, LinearAlgebraError>;
type LuFactorizationResult<'a> = Result<&'a mut Array2<f64>, LinearAlgebraError>;
type QrDecompositionResult = Result<(RealMatrix, RealMatrix), LinearAlgebraError>;

/// A struct representing a matrix of real numbers. The RealMatrix struct is a wrapper around
/// the ndarray::Array2 type, which is a two-dimensional array that is optimized for numerical
/// computations. The RealMatrix struct provides a more user-friendly interface for working with
/// the subset of ndarray's functionality that is needed for this project.
///
/// The RealMatrix struct is used to represent the x and y matrices in the linear regression model,
/// and is the primary data structure used to store the data for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct RealMatrix {
    pub values: Array2<f64>,
}

impl RealMatrix {
    /// Create a new RealMatrix instance from an ndarray::Array2.
    pub fn new(data: Array2<f64>) -> Self {
        RealMatrix { values: data }
    }

    /// Create a new RealMatrix instance with the specified number of rows and columns.
    pub fn with_shape(n_rows: usize, n_cols: usize) -> Self {
        RealMatrix {
            values: Array2::<f64>::zeros((n_rows, n_cols)),
        }
    }

    pub fn is_square(&self) -> bool {
        self.n_rows() == self.n_cols()
    }

    pub fn is_not_square(&self) -> bool {
        !self.is_square()
    }

    /// Create a mutable reference to the values underlying the RealMatrix instance, so that they
    /// can be modified in place.
    pub fn as_mut(&mut self) -> &mut Array2<f64> {
        &mut self.values
    }

    /// Create a new RealMatrix instance from the dot product of two RealMatrix references.
    pub fn dot(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: self.values.dot(&other.values),
        }
    }

    /// Create a new RealMatrix instance from the transpose of the current RealMatrix reference.
    pub fn transpose(&self) -> Result<RealMatrix, LinearAlgebraError> {
        let transposed = self.values.t().to_owned();

        Ok(RealMatrix { values: transposed })
    }

    /// Create a new RealMatrix instance from the element-wise addition of two RealMatrix references.
    pub fn plus(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: &self.values + &other.values,
        }
    }

    /// Create a new RealMatrix instance from the element-wise subtraction of two RealMatrix references.
    pub fn minus(&self, other: &RealMatrix) -> RealMatrix {
        RealMatrix {
            values: &self.values - &other.values,
        }
    }

    /// Create a new RealMatrix instance from a vector of f64 values, coerced into a 2D array with the
    /// specified number of rows and columns.
    pub fn from_vec(data: Vec<f64>, n_rows: usize, n_cols: Option<usize>) -> Self {
        RealMatrix {
            values: Array2::<f64>::from_shape_vec((n_rows, n_cols.unwrap_or(1)), data).unwrap(),
        }
    }

    /// Create an array slice from the RealMatrix instance.
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.values.as_slice()
    }

    /// Create a mutable array slice from the RealMatrix instance.
    pub fn as_slice_mut(&mut self) -> Option<&mut [f64]> {
        self.values.as_slice_mut()
    }

    // Return a two-element array representing the shape of the matrix.
    pub fn shape(&self) -> &[usize; 2] {
        self.values
            .shape()
            .try_into()
            .expect("Shape should have exactly 2 elements")
    }

    /// Return the number of rows in the matrix.
    pub fn n_rows(&self) -> usize {
        self.values.shape()[0]
    }

    /// Return the number of columns in the matrix.
    pub fn n_cols(&self) -> usize {
        self.values.shape()[1]
    }

    /// Return true if the matrix is a vector (i.e. has only one row or one column).
    pub fn is_vec(&self) -> bool {
        self.n_cols() == 1 || self.n_rows() == 1
    }

    /// Return true if the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return the number of dimensions in the matrix. This should always be 2.
    pub fn ndim(&self) -> usize {
        self.values.ndim()
    }

    pub fn inv(&self) -> MatrixInversionResult {
        let mut values = self.values.clone();

        values = self.lu_factorize(&mut values);
        self.invert_from_lu(&mut values)
    }

    fn lu_factorize<'a>(&self, values: &'a mut Array2<f64>) -> LuFactorizationResult<'a> {
        let n_rows = values.nrows() as i32;
        let mut pivot_indices = vec![0; n_rows as usize];
        let mut info = 0;

        unsafe {
            dgetrf(
                n_rows,
                n_rows,
                values.as_slice_mut().unwrap(),
                n_rows,
                &mut pivot_indices,
                &mut info,
            );
        }

        if info != 0 {
            return Err(LinearAlgebraError::MatrixInverseError(
                "LU factorization failed".to_string(),
            ));
        }
        Ok(values)
    }

    fn invert_from_lu(&self, values: &mut Array2<f64>) -> MatrixInversionResult {
        let n = values.nrows() as i32;
        let mut ipiv = vec![0; n as usize];
        let mut info = 0;
        let mut work = vec![0.0; (n as usize) * 64];
        let lwork = work.len() as i32;

        unsafe {
            dgetri(
                n,
                values.as_slice_mut().unwrap(),
                n,
                &mut ipiv,
                &mut work,
                lwork,
                &mut info,
            );
        }

        if info != 0 {
            return Err(LinearAlgebraError::MatrixInverseError(
                "Matrix inversion failed".to_string(),
            ));
        }

        Ok(RealMatrix::new(values.clone()))
    }

    /// Return the QR decomposition of the matrix as two RealMatrix instances.
    pub fn qr(&self) -> QrDecompositionResult {
        let (mut q_values, tau) = self.qr_factorize()?;
        let r_values = self.extract_r(&q_values);
        self.generate_q(&mut q_values, &tau)?;
        Ok((RealMatrix::new(q_values), RealMatrix::new(r_values)))
    }

    fn qr_factorize(&self) -> Result<(Array2<f64>, Vec<f64>), LinearAlgebraError> {
        let (m, n) = self.values.dim();
        let min_mn = m.min(n) as i32;

        let mut a = self.values.clone();
        let mut tau = vec![0.0; min_mn as usize];
        let mut info = 0;

        unsafe {
            let mut work = vec![0.0; 1];
            let mut lwork = -1;

            dgeqrf(
                m as i32,
                n as i32,
                a.as_slice_mut().unwrap(),
                m as i32,
                &mut tau,
                &mut work,
                lwork,
                &mut info,
            );

            lwork = work[0] as i32;
            work = vec![0.0; lwork as usize];

            dgeqrf(
                m as i32,
                n as i32,
                a.as_slice_mut().unwrap(),
                m as i32,
                &mut tau,
                &mut work,
                lwork,
                &mut info,
            );
        }

        if info != 0 {
            return Err(LinearAlgebraError::QrDecompositionError(
                "QR factorization failed".to_string(),
            ));
        }

        Ok((a, tau))
    }

    fn extract_r(&self, a: &Array2<f64>) -> Array2<f64> {
        let (m, n) = a.dim();
        let mut r = Array2::<f64>::zeros((m, n));
        // TODO: Use parallel iterator to fill R matrix more efficiently
        r.axis_iter_mut(ndarray::Axis(0))
            .into_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in i..n {
                    row[j] = a[[i, j]];
                }
            });
        r
    }

    /// Generate the Q matrix from the factored matrix A and tau values.
    fn generate_q(&self, a: &mut Array2<f64>, tau: &[f64]) -> Result<(), LinearAlgebraError> {
        let (m, n) = a.dim();
        let min_mn = m.min(n) as i32;
        let mut info = 0;

        unsafe {
            let mut work = vec![0.0; 1];
            let mut lwork = -1;

            dorgqr(
                m as i32,
                n as i32,
                min_mn,
                a.as_slice_mut().unwrap(),
                m as i32,
                tau,
                &mut work,
                lwork,
                &mut info,
            );

            lwork = work[0] as i32;
            work = vec![0.0; lwork as usize];

            dorgqr(
                m as i32,
                n as i32,
                min_mn,
                a.as_slice_mut().unwrap(),
                m as i32,
                tau,
                &mut work,
                lwork,
                &mut info,
            );
        }

        if info != 0 {
            return Err(LinearAlgebraError::QrDecompositionError(
                "Generating Q matrix failed".to_string(),
            ));
        }

        Ok(())
    }
}
