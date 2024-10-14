// src/traits/mod.rs
pub mod fitter;
pub mod loss_function;
pub mod model;
pub mod predictor;
pub mod scorer;

pub use fitter::{Fitter, FitterReturn};
pub use loss_function::LossFunction;
pub use model::Model;
pub use predictor::Predictor;
pub use scorer::Scorer;
