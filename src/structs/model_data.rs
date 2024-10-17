// src/structs/model_data.rs

use crate::structs::RealMatrix;

/// A struct representing the data for a linear regression model. This struct always maintains
/// ownership of the data, and is used to pass the data safely between functions. This is
/// the only copy of the data that is passed around, and it is never modified.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelData<'a> {
    pub x: &'a RealMatrix,
    pub y: &'a RealMatrix,
}

impl<'a> ModelData<'a> {
    /// Create a new `Data` struct.
    pub fn new(x: &'a RealMatrix, y: &'a RealMatrix) -> Self {
        ModelData { x, y }
    }

    /// Return a reference to the x matrix.
    pub fn x(&self) -> &RealMatrix {
        self.x
    }

    /// Return a reference to the y matrix.
    pub fn y(&self) -> &RealMatrix {
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::create_real_matrix;

    #[test]
    fn test_model_data() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = create_real_matrix(x_values, 2, 3);
        let y = create_real_matrix(y_values, 2, 3);
        let data = ModelData::new(&x, &y);

        assert_eq!(data.x(), &x);
        assert_eq!(data.y(), &y);
    }

    #[test]
    fn test_data_struct_creation() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = create_real_matrix(x_values, 2, 3);
        let y = create_real_matrix(y_values, 2, 3);
        let data = ModelData { x: &x, y: &y };

        assert_eq!(*data.x, x);
        assert_eq!(*data.y, y);
    }
}
