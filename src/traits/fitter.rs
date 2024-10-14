use crate::errors::FittingError;
use crate::structs::ModelData;
use crate::traits::Predictor;

pub type FitterReturn<'a> = Result<Box<(dyn Predictor + 'a)>, FittingError>;
pub trait Fitter<'a> {
    fn fit(&self, data: &'a ModelData) -> FitterReturn;
}
