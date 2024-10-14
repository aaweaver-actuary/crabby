// src/real_matrix.rs

use crate::errors::LinearAlgebraError;
use ndarray::Array2;
use ndarray_linalg::{Inverse, QR};

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

    /// Return the inverse of the matrix as a new RealMatrix instance.
    pub fn inv(&self) -> Result<RealMatrix, LinearAlgebraError> {
        let inv = self
            .values
            .inv()
            .map_err(|e| LinearAlgebraError::MatrixInverseError(e))?;

        Ok(RealMatrix::new(inv))
    }

    /// Return the QR decomposition of the matrix as two RealMatrix instances.
    pub fn qr(&self) -> Result<(RealMatrix, RealMatrix), LinearAlgebraError> {
        let (q, r) = self
            .values
            .qr()
            .map_err(|e| LinearAlgebraError::QrDecompositionError(e))?;

        Ok((RealMatrix::new(q), RealMatrix::new(r)))
    }
}
