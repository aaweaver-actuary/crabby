// src/real_matrix.rs

use crate::prelude::{
    invert_matrix, multiply_matrices, HasLenMethod, MatrixInversionResult,
    MatrixMultiplicationResult,
};
use ndarray::iter::AxisIter;
use ndarray::{s, Array2, Axis, ShapeBuilder};
use std::convert::TryInto;

pub type RowIterator<'a> = AxisIter<'a, f64, ndarray::Dim<[usize; 1]>>;
pub type ColumnIterator<'a> = AxisIter<'a, f64, ndarray::Dim<[usize; 1]>>;

/// Utility function to create a new RealMatrix instance from a vector of f64 values.
///
/// # Arguments
///
/// * `values` - A vector of f64 values to be used as the data for the matrix
/// * `rows` - The number of rows in the matrix
/// * `cols` - The number of columns in the matrix
///
/// # Returns
///
/// A new RealMatrix instance with the specified data, number of rows, and number of columns.
///
/// # Notes
///
/// The length of the `values` vector must be equal to `rows * cols`. In particular, passing
/// a vector of length 5 with `rows=2` and `cols=2` will result in an error. Sparse matrices
/// are not supported.
///
/// # Example
///
/// ```
/// use crabby::prelude::create_real_matrix;
///
/// let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
///
/// assert_eq!(matrix.n_rows(), 2);
/// assert_eq!(matrix.n_cols(), 3);
///
/// let sum_of_values = matrix.values.iter().sum::<f64>();
/// assert_eq!(sum_of_values, 21.0);
/// ```
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

    /// Return a boolean indicating whether the matrix is square.
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

    /// Returns the result as a new RealMatrix.
    pub fn multiply_matrices(a: &mut RealMatrix, b: &mut RealMatrix) -> RealMatrix {
        multiply_matrices(a, b).unwrap()
    }

    /// Create a new RealMatrix instance from the transpose of the current RealMatrix reference.
    pub fn transpose(&self) -> RealMatrix {
        RealMatrix {
            values: self.values.t().into_owned(),
        }
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

    /// Create a new RealMatrix instance from the dot product of two RealMatrix references.
    pub fn dot(&self, other: &RealMatrix) -> MatrixMultiplicationResult {
        let dot_product = self.values.dot(&other.values);

        Ok(RealMatrix {
            values: dot_product,
        })
    }

    /// Calculate the number of columns needed for the array, given the data and the number of rows
    fn get_n_cols<T: HasLenMethod>(data: T, n_rows: usize) -> usize {
        let len = data.len();
        (len + n_rows - 1) / n_rows
    }

    /// Create a new RealMatrix instance from a vector of f64 values, coerced into a 2D array with the
    /// specified number of rows and columns.
    pub fn from_vec(data: Vec<f64>, n_rows: usize, n_cols: Option<usize>) -> Self {
        let n_cols_if_not_provided = self::RealMatrix::get_n_cols(&data, n_rows);

        RealMatrix {
            values: Array2::<f64>::from_shape_vec(
                (n_rows, n_cols.unwrap_or(n_cols_if_not_provided)),
                data,
            )
            .expect("Invalid shape"),
        }
    }

    /// Create a new RealMatrix instance from a vector of f64 values, coerced into a 2D array with the
    /// specified number of rows and columns.
    pub fn from_slice(data: &[f64], n_rows: usize, n_cols: Option<usize>) -> Self {
        let n_cols_if_not_provided = self::RealMatrix::get_n_cols(data, n_rows);

        RealMatrix {
            values: Array2::from_shape_vec(
                (n_rows, n_cols.unwrap_or(n_cols_if_not_provided)),
                data.to_vec(),
            )
            .expect("Invalid shape"),
        }
    }

    /// Return a boolean indicating whether the matrix is in column-major order.
    pub fn is_column_major(&self) -> bool {
        let stride = self.values.strides();
        (stride[0] < stride[1]) && (stride[0] == 1)
    }

    /// Update the RealMatrix instance to be in column-major order, rather than row-major order,
    /// which is the default for ndarray::Array2, overwriting the existing values in the matrix.
    pub fn to_column_major(&mut self) -> Result<(), String> {
        // If the matrix is already in column-major order, return an error
        if self.is_column_major() {
            return Err("Matrix is already in column-major order".to_string());
        }

        // let mut col_major_array = self.allocate_empty_column_major_array();
        let data = self.values.iter().cloned().collect::<Vec<f64>>();
        self.values = Array2::from_shape_vec((self.n_rows(), self.n_cols()).f(), data).unwrap();

        /*         // Fill the column-major array using memory-efficient traversa
        for i in 0..self.n_rows() {
            for j in 0..self.n_cols() {
                // Assign the element at row i, column j from the row-major array
                col_major_array[(i, j)] = self.values[(i, j)];
            }
        }

        self.values = col_major_array; */
        Ok(())
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

    /// Return a result with a reference to the matrix values
    pub fn inv(&mut self) -> MatrixInversionResult {
        let inverted = invert_matrix(self)?;

        Ok(inverted)
    }

    /// Return a result with a vector of the f64 values from the specified row of the matrix.
    pub fn get_row(&self, row: usize) -> Result<Vec<f64>, String> {
        Ok(self.values.slice(s![row, ..]).to_vec())
    }

    /// Return a result with a vector of the f64 values from the specified column of the matrix.
    pub fn get_col(&self, col: usize) -> Result<Vec<f64>, String> {
        Ok(self.values.slice(s![.., col]).to_vec())
    }

    /// Return an iterator over the rows of the matrix.
    pub fn iter_rows(&self) -> RowIterator {
        self.values.axis_iter(Axis(0))
    }

    /// Return an iterator over the columns of the matrix.
    pub fn iter_cols(&self) -> ColumnIterator {
        self.values.axis_iter(Axis(1))
    }
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

    use super::*;
    use ndarray::{array, ShapeBuilder};

    /// Helper function to create a simple RealMatrix for testing
    fn create_simple_matrix() -> RealMatrix {
        RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0]],
        }
    }

    fn get_zeros_vec_to_fill_matrix(matrix: RealMatrix) -> Vec<f64> {
        vec![0.0; matrix.n_rows() * matrix.n_cols()]
    }

    fn allocate_empty_column_major_array(matrix: RealMatrix) -> Array2<f64> {
        let data = get_zeros_vec_to_fill_matrix(matrix.clone());
        Array2::from_shape_vec(
            (matrix.n_rows(), matrix.n_cols()).strides((1, matrix.n_rows())),
            data,
        )
        .unwrap()
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

    #[test]
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

        assert!(!is_square);
    }

    #[test]
    fn test_matrix_is_not_square_when_square() {
        let matrix = create_simple_matrix();
        let is_not_square = matrix.is_not_square();

        assert!(!is_not_square);
    }

    #[test]
    fn test_matrix_is_not_square_when_not_square() {
        let new_matrix = RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        let is_not_square = new_matrix.is_not_square();

        assert!(is_not_square);
    }

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
        let matrix_a = create_simple_matrix();
        let matrix_b = create_simple_matrix();
        let result = matrix_a.dot(&matrix_b);

        assert!(result.is_ok());
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
        let transposed = matrix.transpose();

        assert_eq!(transposed.values, array![[1.0, 3.0], [2.0, 4.0]]);
    }

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
        assert_eq!(matrix.values, array![[1.0, 2.0], [3.0, 4.0]]);

        matrix.to_column_major().unwrap();
        assert_eq!(matrix.values.strides(), [1, 2]);
    }

    #[test]
    fn test_that_to_column_major_doesnt_do_anything_if_the_matrix_is_already_column_major() {
        let mut matrix = create_simple_matrix();

        matrix
            .to_column_major()
            .expect("Failed to convert matrix to column-major order (first time)");

        let mut expected_column_major_matrix = allocate_empty_column_major_array(matrix.clone());

        for i in 0..matrix.n_rows() {
            for j in 0..matrix.n_cols() {
                expected_column_major_matrix[(i, j)] = matrix.values[(i, j)];
            }
        }

        assert_eq!(matrix.values, expected_column_major_matrix);
        assert!(matrix.to_column_major().is_err());
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

    #[test]
    fn test_create_real_matrix() {
        let real_matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let expected_matrix = RealMatrix {
            values: array![[1.0, 2.0], [3.0, 4.0]],
        };

        assert_eq!(real_matrix, expected_matrix);
    }

    #[test]
    fn test_create_4_by_7_real_matrix() {
        let real_matrix = create_real_matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
            ],
            4,
            7,
        );

        let expected_matrix = RealMatrix {
            values: array![
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
            ],
        };

        assert_eq!(real_matrix, expected_matrix);
    }

    #[test]
    fn test_as_mut_array_ref() {
        let mut matrix = create_simple_matrix();
        let matrix_clone = matrix.clone();
        let matrix_ref = matrix.as_mut_array_ref();

        assert_eq!(matrix_ref, &matrix_clone);
    }

    #[test]
    fn test_creation_of_real_matrix_success() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let real_matrix = RealMatrix {
            values: data.clone(),
        };

        assert_eq!(real_matrix.values, data);
    }

    #[test]
    fn test_as_mut_array_ref_pt2() {
        let mut matrix = create_simple_matrix();
        let matrix_clone = matrix.clone();
        let matrix_ref = matrix.as_mut_array_ref();

        assert_eq!(matrix_ref, &matrix_clone);
    }

    #[test]
    fn test_from_array2_for_real_matrix() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let real_matrix: RealMatrix = data.clone().into();
        assert_eq!(real_matrix.values, data);
    }

    #[test]
    fn test_from_real_matrix_for_array2() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let real_matrix = RealMatrix {
            values: data.clone(),
        };
        let array2: Array2<f64> = real_matrix.into();

        assert_eq!(array2, data);
    }

    #[test]
    fn test_as_ref_for_real_matrix() {
        let matrix = create_simple_matrix();
        let matrix_ref = matrix.as_ref();

        assert_eq!(matrix_ref, &matrix);
    }

    #[test]
    fn test_as_mut_for_real_matrix() {
        let mut matrix = create_simple_matrix();
        {
            let matrix_mut = matrix.as_mut();
            assert_eq!(matrix_mut as *mut _, &mut matrix as *mut _);
        }
    }

    #[test]
    fn test_as_slice_creates_the_expected_option_return() {
        let matrix = create_simple_matrix();
        let slice_option = matrix.as_slice();

        assert_eq!(slice_option, Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_as_slice_unwraps_successfully() {
        let matrix = create_simple_matrix();
        let slice = matrix.as_slice().unwrap();

        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_as_slice_mut_creates_the_expected_option_return() {
        let mut matrix = create_simple_matrix();
        let slice_option = matrix.as_slice_mut();

        assert_eq!(slice_option, Some(&mut [1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_as_slice_mut_creates_a_mutable_object() {
        let mut matrix = create_simple_matrix();
        let slice = matrix.as_slice_mut();

        assert_eq!(slice, Some(&mut [1.0, 2.0, 3.0, 4.0][..]));

        let mut matrix = create_simple_matrix();
        let slice = matrix.as_slice_mut().unwrap();

        // show it will not fail if we try to change the values
        slice[0] = 5.0;
        slice[1] = 6.0;
        slice[2] = 7.0;
        slice[3] = 8.0;

        assert_eq!(slice[0], 5.0);
        assert_eq!(slice[1], 6.0);
        assert_eq!(slice[2], 7.0);
        assert_eq!(slice[3], 8.0);
    }

    #[test]
    fn test_inverse() {
        let mut matrix = create_simple_matrix();
        let inverted = matrix.inv().unwrap();

        let expected = create_real_matrix(vec![-2.0, 1.0, 1.5, -0.5], 2, 2);

        assert_eq!(inverted.values, expected.values);
    }

    #[test]
    fn test_is_column_major() {
        let matrix = RealMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, Some(2));

        // if it is column major, the matrix will be:
        // [[1.0, 3.0],
        //  [2.0, 4.0]]

        let expected = matrix.values[[0, 0]] == 1.0
            && matrix.values[[0, 1]] == 3.0
            && matrix.values[[1, 0]] == 2.0
            && matrix.values[[1, 1]] == 4.0;
        let actual = matrix.is_column_major();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_get_zeros_vec_to_fill_matrix() {
        let matrix = create_simple_matrix();
        let zeros = get_zeros_vec_to_fill_matrix(matrix);

        assert_eq!(zeros, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_allocate_empty_column_major_array() {
        let matrix = create_simple_matrix();
        let empty_column_major_array = allocate_empty_column_major_array(matrix);

        assert_eq!(empty_column_major_array, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_empty_column_major_array_is_actually_column_major() {
        let matrix = create_simple_matrix();
        let empty_column_major_array = allocate_empty_column_major_array(matrix);

        let expected = empty_column_major_array.strides();
        let actual = [1, 2];

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_matrix_multiplication() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix_b = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let matrix_ab = RealMatrix::multiply_matrices(&mut matrix_a.clone(), &mut matrix_b.clone());

        let expected = create_real_matrix(vec![7.0, 10.0, 15.0, 22.0], 2, 2);

        assert_eq!(matrix_ab.values, expected.values);
        assert_eq!(matrix_ab.n_rows(), 2);
        assert_eq!(matrix_ab.n_cols(), 2);
    }

    #[test]
    fn test_get_row() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        assert_eq!(matrix.get_row(0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(matrix.get_row(1).unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn test_get_row_out_of_bounds() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let result = matrix.get_row(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_row_returns_a_copy() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let row = matrix.get_row(0).unwrap();
        let row_copy = row.clone();

        assert_eq!(row, row_copy);
    }

    #[test]
    fn test_get_col() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        assert_eq!(matrix.get_col(0).unwrap(), vec![1.0, 3.0]);
        assert_eq!(matrix.get_col(1).unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn test_get_col_out_of_bounds() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let result = matrix.get_col(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_row_iterator() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        // Compute the sum of squares of each row
        let row_iter = matrix
            .iter_rows()
            .map(|row| row.map(|x| x.powi(2)).sum())
            .collect::<Vec<f64>>();

        let expected = vec![5.0, 25.0];

        assert_eq!(row_iter, expected);
    }

    #[test]
    fn test_col_iterator() {
        let matrix = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        // Compute the sum of squares of each column
        let col_iter = matrix
            .iter_cols()
            .map(|col| col.map(|x| x.powi(2)).sum())
            .collect::<Vec<f64>>();

        let expected = vec![10.0, 20.0];

        assert_eq!(col_iter, expected);
    }
}
