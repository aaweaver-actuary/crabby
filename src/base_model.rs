use crate::structs::ModelData;
use derive_builder::Builder;

#[derive(Debug, Builder)]
pub struct BaseModel<'a> {
    data: &'a ModelData,
    
}
