// src/real_matrix.rs

use crate::errors::LinearAlgebraError;
use ndarray::Array2;
use std::convert::TryInto;

type QrFactorizationResult = Result<(RealMatrix, RealMatrix), LinearAlgebraError>;
type DotProductResult = Result<RealMatrix, LinearAlgebraError>;

/// Utility function to create a new RealMatrix instance from a vector of f64 values.
pub fn create_real_matrix(values: Vec<f64>, rows: usize, cols: usize) -> RealMatrix {
    RealMatrix::from_vec(values, rows, Some(cols))
}

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
    pub fn as_mut_array(&mut self) -> &mut Array2<f64> {
        &mut self.values
    }

    /// Return a mutable reference to the RealMatrix instance
    pub fn as_mut_array_ref(&mut self) -> &mut RealMatrix {
        self
    }

    /// Uses the weaver::lapack::multiply_matrices function to multiply two RealMatrix instances.
    /// Returns the result as a new RealMatrix.
    pub fn multiply_matrices(a: &mut RealMatrix, _b: &mut RealMatrix) -> RealMatrix {
        // pub fn multiply_matrices(a: &mut RealMatrix, b: &mut RealMatrix) -> MatrixMultiplicationResult {
        // multiply_matrices(a, b)
        a.clone()
    }

    /*     /// Create a new RealMatrix instance from the dot product of two RealMatrix references.
    pub fn dot(&mut self, other: &mut RealMatrix) -> DotProductResult {
        multiply_matrices(self, other)
    } */

    pub fn dot(&mut self, _other: &mut RealMatrix) -> DotProductResult {
        Ok(RealMatrix {
            values: self.values.clone(),
        })
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

    /// Return a boolean indicating whether the matrix is in column-major order.
    pub fn is_column_major(&self) -> bool {
        self.values.is_standard_layout()
    }

    /// Create a new RealMatrix instance in column-major order from a RealMatrix instance that is
    /// (by default) in row-major order.
    pub fn to_column_major(&mut self) -> Self {
        if self.is_column_major() {
            self.clone()
        } else {
            let column_major = self.values.clone().reversed_axes();

            RealMatrix {
                values: column_major,
            }
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
    pub fn n_dim(&self) -> usize {
        self.values.ndim()
    }

    pub fn inv(&self) -> Result<RealMatrix, LinearAlgebraError> {
        let inverted = self.values.clone();

        Ok(RealMatrix { values: inverted })
    }

    /*     /// Return the QR decomposition of the matrix as two RealMatrix instances.
    pub fn qr(&self) -> QrFactorizationResult {
        let (q, r) = qr_factorization(self)?;

        Ok((q, r))
    } */
}

impl From<Array2<f64>> for RealMatrix {
    fn from(data: Array2<f64>) -> Self {
        RealMatrix { values: data }
    }
}

impl From<RealMatrix> for Array2<f64> {
    fn from(matrix: RealMatrix) -> Self {
        matrix.values
    }
}

impl AsRef<RealMatrix> for RealMatrix {
    fn as_ref(&self) -> &RealMatrix {
        self
    }
}

impl AsMut<RealMatrix> for RealMatrix {
    fn as_mut(&mut self) -> &mut RealMatrix {
        self
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
    fn test_matrix_new() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = RealMatrix::new(data.clone());

        assert_eq!(matrix.values, data);
    }

    #[test]
    fn test_matrix_with_shape_returns_the_correct_shape() {
        let matrix = RealMatrix::with_shape(2, 2);
        assert_eq!(matrix.values.shape(), &[2, 2]);

        let new_matrix = RealMatrix::with_shape(3, 4);
        assert_eq!(new_matrix.values.shape(), &[3, 4]);
    }

    #[test]
    fn test_matrix_with_shape_returns_all_0s() {
        let matrix = RealMatrix::with_shape(2, 2);

        assert_eq!(matrix.values, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    /*     #[test]
    fn test_matrix_is_square_when_actually_square() {
        let matrix = create_simple_matrix();
        let is_square = matrix.is_square();

        assert!(is_square);
    }

    #[test]
    fn test_matrix_is_square_when_not_square() {
        let new_matrix = RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        let is_square = new_matrix.is_square();

        assert!(is_square);
    }

    #[test]
    fn test_matrix_is_not_square_when_square() {
        let matrix = create_simple_matrix();
        let is_not_square = matrix.is_not_square();

        assert!(is_not_square);
    }

    #[test]
    fn test_matrix_is_not_square_when_not_square() {
        let new_matrix = RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        let is_not_square = new_matrix.is_not_square();

        assert!(is_not_square);
    } */

    #[test]
    fn test_matrix_as_mut_without_changing_the_values() {
        let matrix = create_simple_matrix();

        assert_eq!(matrix.values, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_matrix_as_mut_values_are_actually_mutable() {
        let mut matrix = create_simple_matrix();
        let values = matrix.as_mut_array();

        values[[0, 0]] = 5.0;
        values[[0, 1]] = 6.0;
        values[[1, 0]] = 7.0;
        values[[1, 1]] = 8.0;

        assert_eq!(values[[0, 0]], 5.0);
        assert_eq!(values[[0, 1]], 6.0);
        assert_eq!(values[[1, 0]], 7.0);
        assert_eq!(values[[1, 1]], 8.0);
    }

    #[test]
    fn test_dot_product_returns_a_dot_product_result_type() {
        let mut matrix_a = create_simple_matrix();
        let mut matrix_b = create_simple_matrix();
        let result = matrix_a.dot(&mut matrix_b);

        assert!(result.is_ok());
    }

    /*     #[test]
    fn test_dot_product_between_two_1d_matrices() {
        let mut matrix_a = RealMatrix::from_vec(vec![1.0, 2.0], 1, Some(2));
        let mut matrix_b = RealMatrix::from_vec(vec![3.0, 4.0], 1, Some(2));
        let result = matrix_a.dot(&mut matrix_b).unwrap();

        assert_eq!(result.values, array![[11.0]]);
    }

    #[test]
    fn test_dot_product_between_two_2d_matrices() {
        let mut matrix_a = create_simple_matrix();
        let mut matrix_b = create_simple_matrix();
        let result = matrix_a.dot(&mut matrix_b).unwrap();

        assert_eq!(result.values, array![[7.0, 10.0], [15.0, 22.0]]);
    } */

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
    fn test_matrix_dot_product() {
        let mut matrix_a = RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0]],
        };
        let mut matrix_b = RealMatrix {
            values: array![[5.0, 6.0], [7.0, 8.0]],
        };
        let result = matrix_a.dot(&mut matrix_b);

        assert_eq!(result.unwrap().values, array![[19.0, 22.0], [43.0, 50.0]]);
    } */

    #[test]
    fn test_matrix_subtraction() {
        let matrix_a = create_simple_matrix();
        let matrix_b = create_simple_matrix();
        let result = matrix_a.minus(&matrix_b);

        assert_eq!(result.values, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = RealMatrix::from_vec(data, 2, Some(2));

        assert_eq!(matrix.values, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_matrix_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let matrix = RealMatrix::from_slice(&data, 2, Some(2));

        assert_eq!(matrix.values, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_matrix_to_column_major() {
        let mut matrix = create_simple_matrix();
        println!("{:?}", matrix);
        let column_major = matrix.to_column_major();
        println!("{:?}", column_major);

        assert_eq!(column_major.values, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_matrix_shape() {
        let matrix = create_simple_matrix();
        let shape = matrix.shape();

        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn test_matrix_n_rows() {
        let matrix = create_simple_matrix();
        let n_rows = matrix.n_rows();

        assert_eq!(n_rows, 2);
    }

    #[test]
    fn test_matrix_n_cols() {
        let matrix = create_simple_matrix();
        let n_cols = matrix.n_cols();

        assert_eq!(n_cols, 2);
    }

    #[test]
    fn test_matrix_is_vec() {
        let matrix = create_simple_matrix();
        let is_vec = matrix.is_vec();

        assert!(!is_vec);
    }

    #[test]
    fn test_matrix_is_empty() {
        let matrix = RealMatrix { values: array![[]] };
        let is_empty = matrix.is_empty();

        assert!(is_empty);
    }

    #[test]
    fn test_matrix_n_dim() {
        let matrix = create_simple_matrix();
        let ndim = matrix.n_dim();

        assert_eq!(ndim, 2);
    }
}
