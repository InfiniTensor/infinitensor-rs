use common::{invoke_ty, Data, DataType};
use std::{
    collections::HashMap,
    fmt,
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::Tensor;

#[allow(unused)]
pub struct TensorPos {
    pub(super) op: usize,
    pub(super) idx: usize,
}

pub struct LinkedTensor {
    tensor: Tensor,
    pub(super) source: Mutex<HashMap<usize, TensorPos>>,
    pub(super) target: Mutex<HashMap<usize, Vec<TensorPos>>>,
}

impl Deref for LinkedTensor {
    type Target = Tensor;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl LinkedTensor {
    #[inline]
    pub fn share(shape: Vec<usize>, dtype: DataType, data: Option<Arc<Data>>) -> Arc<Self> {
        Arc::new(Self {
            tensor: Tensor { shape, dtype, data },
            source: Default::default(),
            target: Default::default(),
        })
    }
}

impl fmt::Display for LinkedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}<{}>{{{}}}",
            self.tensor.dtype,
            self.tensor
                .shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x"),
            if let Some(data) = self.tensor.data() {
                invoke_ty!(self.tensor.dtype, data.as_slice(), data_to_string)
            } else {
                String::new()
            },
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
