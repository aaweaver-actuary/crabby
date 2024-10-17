use crate::errors::FittingError;

pub type FitterReturn = Result<(), FittingError>;
pub trait Fitter<'a> {
    fn fit(&mut self) -> FitterReturn;
}
