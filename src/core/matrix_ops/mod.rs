pub mod blas_transpose_flag;
pub mod least_squares_solver;
pub mod matrix_inverter;
pub mod matrix_multiplier;

pub use blas_transpose_flag::BlasTransposeFlag;
pub use least_squares_solver::{solve_least_squares, LeastSquaresResult};
pub use matrix_inverter::{invert_matrix, MatrixInversionResult};
pub use matrix_multiplier::{multiply_matrices, MatrixMultiplicationResult};
