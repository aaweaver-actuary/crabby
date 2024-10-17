pub mod errors;
pub mod lapack_;
pub mod plugins;
pub mod structs;
pub mod traits;

pub use lapack_::invert_matrix;
pub use structs::{create_real_matrix, RealMatrix};
