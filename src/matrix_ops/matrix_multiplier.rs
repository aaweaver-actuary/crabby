use crate::prelude::{create_real_matrix, errors::LinearAlgebraError, RealMatrix};

pub fn multiply_matrices(
    matrix_a: &RealMatrix,
    matrix_b: &RealMatrix,
) -> Result<RealMatrix, LinearAlgebraError> {
    let multiplier = MatrixMultiplier::new(matrix_a, matrix_b);

    if multiplier.has_dimension_mismatch() {
        panic!("matrix_a.n_cols() != matrix_b.n_rows()");
    }
    multiplier.multiply()
}

struct MatrixMultiplier<'a> {
    matrix_m_n: &'a RealMatrix,
    matrix_n_p: &'a RealMatrix,
    m: usize,
    n: usize,
    p: usize,
}

impl<'a> MatrixMultiplier<'a> {
    fn new(matrix_m_n: &'a RealMatrix, matrix_n_p: &'a RealMatrix) -> Self {
        MatrixMultiplier {
            matrix_m_n,
            matrix_n_p,
            m: matrix_m_n.n_rows(),
            n: matrix_m_n.n_cols(),
            p: matrix_n_p.n_cols(),
        }
    }

    fn has_dimension_mismatch(&self) -> bool {
        self.matrix_m_n.n_cols() != self.matrix_n_p.n_rows()
    }

    fn multiply(&self) -> Result<RealMatrix, LinearAlgebraError> {
        if self.has_dimension_mismatch() {
            panic!("matrix_a.n_cols() != matrix_b.n_rows()");
        }

        let mut result = vec![0.0; self.m * self.p];
        for i in 0..self.m {
            for j in 0..self.p {
                for k in 0..self.n {
                    let a = self.matrix_m_n.values[[i, k]];
                    let b = self.matrix_n_p.values[[k, j]];
                    result[i * self.p + j] += a * b;
                }
            }
        }

        Ok(create_real_matrix(result, self.m, self.p))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_can_use_public_exposed_function() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix_b = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let matrix_ab = multiply_matrices(&matrix_a, &matrix_b).unwrap();

        let expected = create_real_matrix(vec![7.0, 10.0, 15.0, 22.0], 2, 2);

        assert_eq!(matrix_ab, expected);
        assert_eq!(matrix_ab.n_rows(), 2);
        assert_eq!(matrix_ab.n_cols(), 2);
    }

    #[test]
    fn test_can_make_matrix_multiplier() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix_b = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let multiplier = MatrixMultiplier::new(matrix_a.as_ref(), matrix_b.as_ref());

        assert_eq!(multiplier.m, 2);
        assert_eq!(multiplier.n, 2);
        assert_eq!(multiplier.p, 2);

        assert_eq!(multiplier.matrix_m_n, matrix_a.as_ref());
        assert_eq!(multiplier.matrix_n_p, matrix_b.as_ref());
    }

    #[test]
    fn test_matrix_multiplier_multiply_method() {
        let matrix_a = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix_b = create_real_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let multiplier = MatrixMultiplier::new(matrix_a.as_ref(), matrix_b.as_ref());

        let result = multiplier.multiply().unwrap();

        let expected = create_real_matrix(vec![7.0, 10.0, 15.0, 22.0], 2, 2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplying_matrices_with_different_dimensions() {
        let arr1 = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
        let arr2 = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let matrix_a = RealMatrix::from(arr1);
        let matrix_b = RealMatrix::from(arr2);

        let result = multiply_matrices(&matrix_a, &matrix_b).unwrap();
        let expected = create_real_matrix(vec![46.0, 52.0, 88.0, 100.0], 2, 2);

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic = "matrix_a.n_cols() != matrix_b.n_rows()"]
    fn test_multiplying_matrices_with_incompatible_dimensions() {
        let arr1 = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        let arr2 = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let matrix_a = RealMatrix::from(arr1);
        let matrix_b = RealMatrix::from(arr2);

        let multiplier = MatrixMultiplier::new(matrix_a.as_ref(), matrix_b.as_ref());

        multiplier.multiply().unwrap();
    }
}
