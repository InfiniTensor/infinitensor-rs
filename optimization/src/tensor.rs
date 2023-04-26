use common::{AsDataType, Data, DataType};
use std::{
    collections::HashMap,
    fmt,
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
    pub fn shape(&self) -> &[usize] {
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

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}<{}>{{{}}}",
            self.data_type,
            self.shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x"),
            match self.data_type {
                DataType::UNDEFINED => unreachable!(),
                DataType::FLOAT => data_to_string::<f32>(&self.data),
                DataType::UINT8 => data_to_string::<u8>(&self.data),
                DataType::INT8 => data_to_string::<i8>(&self.data),
                DataType::UINT16 => data_to_string::<u16>(&self.data),
                DataType::INT16 => data_to_string::<i16>(&self.data),
                DataType::INT32 => data_to_string::<i32>(&self.data),
                DataType::INT64 => data_to_string::<i64>(&self.data),
                DataType::STRING => data_to_string::<char>(&self.data),
                DataType::BOOL => data_to_string::<bool>(&self.data),
                DataType::DOUBLE => data_to_string::<f64>(&self.data),
                DataType::FLOAT16 => todo!(),
                DataType::UINT32 => data_to_string::<u32>(&self.data),
                DataType::UINT64 => data_to_string::<u64>(&self.data),
            },
        )
    }
}

#[inline]
fn data_to_string<T: ToString>(data: &Data) -> String {
    let slice = data.as_slice::<T>();
    if slice.len() > 12 {
        let mut ans = slice[..6]
            .iter()
            .map(T::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        ans.extend(", ...".chars());
        ans
    } else {
        slice
            .iter()
            .map(T::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}
