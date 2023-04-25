use common::{AsDataType, Data, DataType};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub struct TensorPos {
    pub(crate) op: usize,
    pub(crate) idx: usize,
}

pub struct Tensor {
    shape: Vec<usize>,
    data_type: DataType,
    data: Data,
    pub(crate) source: Mutex<HashMap<usize, TensorPos>>,
    pub(crate) target: Mutex<HashMap<usize, Vec<TensorPos>>>,
}

impl Tensor {
    #[inline]
    pub fn share(shape: Vec<usize>, data_type: DataType, data: Data) -> Arc<Self> {
        Arc::new(Self {
            shape,
            data_type,
            data,
            source: Default::default(),
            target: Default::default(),
        })
    }

    #[inline]
    pub fn share_value<T: AsDataType>(val: T) -> Arc<Self> {
        Self::share(vec![1], T::as_data_type(), Data::cpu(vec![val]))
    }

    #[inline]
    pub fn share_vec<T: AsDataType>(vec: Vec<T>) -> Arc<Self> {
        Self::share(vec![vec.len()], T::as_data_type(), Data::cpu(vec))
    }

    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    #[inline]
    pub fn data(&self) -> &Data {
        &self.data
    }

    #[inline]
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.count() * self.data_type.size()
    }
}
