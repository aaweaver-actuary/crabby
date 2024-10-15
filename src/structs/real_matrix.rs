// src/real_matrix.rs

use crate::errors::LinearAlgebraError;
use crate::lapack::qr_factorization;
use ndarray::Array2;

type QrFactorizationResult = Result<(RealMatrix, RealMatrix), LinearAlgebraError>;

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

    /// Create a new RealMatrix instance from a vector of f64 values, coerced into a 2D array with the
    /// specified number of rows and columns.
    pub fn from_slice(data: &[f64], n_rows: usize, n_cols: Option<usize>) -> Self {
        RealMatrix {
            values: Array2::<f64>::from_shape_vec((n_rows, n_cols.unwrap_or(1)), data.to_vec())
                .unwrap(),
        }
    }

    /// Create a new RealMatrix instance in column-major order from a RealMatrix instance that is
    /// (by default) in row-major order.
    pub fn to_column_major(&self) -> RealMatrix {
        RealMatrix {
            values: self.values.clone().reversed_axes(),
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

    pub fn inv(&self) -> Result<RealMatrix, LinearAlgebraError> {
        let inverted = self.values.clone();

        Ok(RealMatrix { values: inverted })
    }
    /*     /// Return the inverse of the matrix as a new RealMatrix instance.
    pub fn inv(&self) -> MatrixInversionResult {
        let result = invert_matrix(&self);

        match result {
            Ok(inverted) => Ok(inverted),
            Err(error) => Err(error),
        }
    } */

    /// Return the QR decomposition of the matrix as two RealMatrix instances.
    pub fn qr(&self) -> QrFactorizationResult {
        let (q, r) = qr_factorization(&self)?;

        Ok((q, r))
    }
}

#[cfg(test)]

mod tests {
    use super::RealMatrix;
    use ndarray::array;

    /// Helper function to create a simple RealMatrix for testing
    fn create_simple_matrix() -> RealMatrix {
        RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0]],
        }
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = create_simple_matrix();
        assert_eq!(matrix.values.shape(), &[2, 2]);
        assert_eq!(matrix.values[[0, 0]], 1.0);
        assert_eq!(matrix.values[[0, 1]], 2.0);
        assert_eq!(matrix.values[[1, 0]], 3.0);
        assert_eq!(matrix.values[[1, 1]], 4.0);
    }

    #[test]
    fn test_matrix_addition() {
        let matrix_a = create_simple_matrix();
        let matrix_b = create_simple_matrix();
        let result = matrix_a.plus(&matrix_b);

        assert_eq!(result.values, array![[2.0, 4.0], [6.0, 8.0]]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = create_simple_matrix();
        let transposed = matrix.transpose().expect("Failed to transpose matrix");

        assert_eq!(transposed.values, array![[1.0, 3.0], [2.0, 4.0]]);
    }

    /*     #[test]
    fn test_matrix_inversion() {
        let matrix = RealMatrix {
            values: array![[4.0, 7.0], [2.0, 6.0]],
        };
        let inverted = matrix.inv().unwrap();

        // Expected result is calculated manually or using a reliable tool
        let expected = array![[0.6, -0.7], [-0.2, 0.4]];
        for i in 0..2 {
            for j in 0..2 {
                assert_approx_eq!(inverted.values[[i, j]], expected[[i, j]], 1e-6);
            }
        }
    } */
}
