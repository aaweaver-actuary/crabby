pub mod core;
pub mod errors;
pub mod matrix_ops;
pub mod plugins;
pub mod structs;
pub mod traits;

pub mod prelude {
    pub use crate::matrix_ops::{invert_matrix, multiply_matrices, solve_least_squares};
    pub use crate::plugins::{fitters, loss, optimizers, predictors, scorers};
    pub use crate::structs::{create_real_matrix, ModelData, RealMatrix};
    pub use crate::traits::{HasLenMethod, LossFunction, Model, Predictor, Scorer};
    pub use crate::errors;
}
