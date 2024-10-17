pub mod errors;
pub mod matrix_ops;
pub mod plugins;
pub mod structs;
pub mod traits;

pub use matrix_ops::invert_matrix;
pub use structs::{create_real_matrix, RealMatrix, ModelData};
