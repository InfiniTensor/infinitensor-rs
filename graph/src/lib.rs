mod graph;
mod infer;

use common::{AsDataType, Data, DataType};
use std::sync::Arc;

pub use graph::dense;

#[derive(Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    dtype: DataType,
    data: Option<Arc<Data>>,
}

impl Tensor {
    #[inline]
    pub fn clone_info(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dtype: self.dtype,
            data: None,
        }
    }

    #[inline]
    pub fn is_variable(&self) -> bool {
        matches!(self.shape.as_slice(), &[1])
    }

    #[inline]
    pub fn is_typed_variable<T: AsDataType>(&self) -> bool {
        self.dtype == T::as_data_type() && matches!(self.shape.as_slice(), &[1])
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size(&self) -> usize {
        self.dtype.size() * self.count()
    }

    #[inline]
    pub fn data(&self) -> &Option<Arc<Data>> {
        &self.data
    }
}
