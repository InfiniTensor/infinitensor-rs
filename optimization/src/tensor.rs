use common::{invoke_ty, AsDataType, Data, DataType};
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

#[allow(unused)]
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
            invoke_ty!(self.data_type, self.data.as_slice(), data_to_string),
        )
    }
}

#[inline]
fn data_to_string<T: ToString>(slice: &[T]) -> String {
    if slice.len() > 12 {
        let mut ans = String::new();
        for x in &slice[..6] {
            ans.push_str(x.to_string().as_str());
            ans.push_str(", ");
        }
        ans.push_str("...");
        ans
    } else {
        slice
            .iter()
            .map(T::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}
