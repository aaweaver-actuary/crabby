#[derive(Debug)]
pub enum ModelError {
    DataError(String),
    FitError(String),
    PredictError(String),
    EvaluationError(String),
}
