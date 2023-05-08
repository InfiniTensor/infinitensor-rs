mod graph;
mod infer;

use common::{invoke_ty, AsDataType, Data, DataType};
use std::{fmt, sync::Arc};

pub use graph::{dense, linked, Graph, Operator};

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
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    #[inline]
    pub fn is_typed_scalar<T: AsDataType>(&self) -> bool {
        self.is_scalar() && self.dtype == T::as_data_type()
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

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}<{}>{{{}}}",
            self.dtype(),
            self.shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x"),
            if let Some(data) = self.data.as_ref() {
                invoke_ty!(self.dtype(), data.as_slice(), data_to_string)
            } else {
                String::new()
            }
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
