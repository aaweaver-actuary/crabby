pub use crate::core;

pub use crate::core::errors;
pub use crate::core::matrix_ops;
pub use crate::core::structs;
pub use crate::core::traits;
pub use crate::plugins;

pub use crate::core::matrix_ops::{
    invert_matrix, multiply_matrices, solve_least_squares, LeastSquaresResult,
    MatrixInversionResult, MatrixMultiplicationResult,
};
pub use crate::core::structs::{create_real_matrix, ModelData, RealMatrix};
pub use crate::core::traits::{HasLenMethod, LossFunction, Model, Predictor, Scorer};
pub use crate::plugins::{fitters, loss, optimizers, predictors, scorers};
