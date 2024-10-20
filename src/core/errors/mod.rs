// src/errors/mod.rs

pub mod fitting_error;
pub mod linear_algebra_error;
pub mod model_error;

pub use fitting_error::FittingError;
pub use linear_algebra_error::LinearAlgebraError;
pub use model_error::ModelError;
