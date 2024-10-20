use ndarray::Array2;

/// Trait for objects that have a length method
pub trait HasLenMethod {
    /// Returns the length of the object
    ///
    /// # Returns
    /// usize - the length of the object
    ///
    /// # Example
    /// ```
    /// use crabby::prelude::HasLenMethod;
    ///
    /// let vector = vec![1.0, 2.0, 3.0];
    /// assert_eq!(vector.len(), 3);
    /// ```
    fn len(&self) -> usize;

    /// Returns true if the object is empty
    ///
    /// # Returns
    /// bool - true if the object is empty
    ///
    /// # Example
    /// ```
    /// use crabby::prelude::HasLenMethod;
    ///
    /// let empty_vec: Vec<f64> = vec![];
    /// assert!(empty_vec.is_empty());
    ///
    /// let nonempty_vec: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// assert!(!nonempty_vec.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/* impl HasLenMethod for Vec<f64> {
    fn len(&self) -> usize {
        self.len()
    }
} */

impl HasLenMethod for [f64] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl HasLenMethod for &[f64] {
    fn len(&self) -> usize {
        let mut output = 0;
        self.iter().for_each(|_| output += 1);
        output as usize
    }
}

impl HasLenMethod for &Vec<f64> {
    fn len(&self) -> usize {
        let mut output = 0;
        self.iter().for_each(|_| output += 1);
        output as usize
    }
}

impl HasLenMethod for Array2<f64> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl HasLenMethod for &Array2<f64> {
    fn len(&self) -> usize {
        let mut output = 0;
        self.iter().for_each(|_| output += 1);
        output as usize
    }
}

#[cfg(test)]

mod tests {

    use ndarray::{array, Array2};

    /*     #[test]
    fn test_vec_len() {
        let vec = vec![1.0, 2.0, 3.0];
        assert_eq!(vec.len(), 3);
    } */

    #[test]
    fn test_vec_ref_len() {
        let vec = vec![1.0, 2.0, 3.0];
        let vec_ref = &vec;
        assert_eq!(vec_ref.len(), 3);
    }

    #[test]
    fn test_array_len() {
        let array = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(array.len(), 4);
    }

    #[test]
    fn test_array_ref_len() {
        let array = array![[1.0, 2.0], [3.0, 4.0]];
        let array_ref = &array;
        assert_eq!(array_ref.len(), 4);
    }

    #[test]
    fn test_slice_len() {
        let slice = &[1.0, 2.0, 3.0];
        assert_eq!(slice.len(), 3);
    }

    #[test]
    fn test_slice_ref_len() {
        let slice = &[1.0, 2.0, 3.0];
        let slice_ref = &slice;
        assert_eq!(slice_ref.len(), 3);
    }

    #[test]
    fn test_empty_vec_is_empty() {
        let vec: Vec<f64> = vec![];
        assert!(vec.is_empty());
    }

    #[test]
    fn test_empty_vec_ref_is_empty() {
        let vec: Vec<f64> = vec![];
        let vec_ref = &vec;
        assert!(vec_ref.is_empty());
    }

    #[test]
    fn test_empty_array_is_empty() {
        let array: Array2<f64> = Array2::zeros((0, 0));
        assert!(array.is_empty());
    }

    #[test]
    fn test_empty_array_ref_is_empty() {
        let array: Array2<f64> = Array2::zeros((0, 0));
        let array_ref = &array;
        assert!(array_ref.is_empty());
    }

    #[test]
    fn test_empty_slice_is_empty() {
        let slice: &[f64] = &[];
        assert!(slice.is_empty());
    }

    #[test]
    fn test_empty_slice_ref_is_empty() {
        let slice: &[f64] = &[];
        let slice_ref = &slice;
        assert!(slice_ref.is_empty());
    }
}
