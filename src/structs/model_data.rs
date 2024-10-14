// src/structs/model_data.rs

use crate::structs::RealMatrix;

/// A struct representing the data for a linear regression model. This struct always maintains
/// ownership of the data, and is used to pass the data safely between functions. This is
/// the only copy of the data that is passed around, and it is never modified.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelData {
    pub x: RealMatrix,
    pub y: RealMatrix,
}

impl ModelData {
    /// Create a new `Data` struct.
    pub fn new(x: RealMatrix, y: RealMatrix) -> Self {
        ModelData { x, y }
    }

    /// Return a reference to the x matrix.
    pub fn x(&self) -> &RealMatrix {
        &self.x
    }

    /// Return a reference to the y matrix.
    pub fn y(&self) -> &RealMatrix {
        &self.y
    }
}
