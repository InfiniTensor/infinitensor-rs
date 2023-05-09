mod convert;
mod graph;
mod linked_tensor;

use super::Operator as OpTrait;
use basic_operator::OpType;
use std::sync::{Arc, Weak};

pub use graph::Unigraph;
pub use linked_tensor::{LinkedTensor, TensorPos};

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct WeakTensor(Weak<LinkedTensor>);

impl PartialEq for WeakTensor {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl Eq for WeakTensor {}

impl PartialOrd for WeakTensor {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.as_ptr().partial_cmp(&other.0.as_ptr())
    }
}

impl Ord for WeakTensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ptr().cmp(&other.0.as_ptr())
    }
}

pub struct Operator {
    op_type: OpType,
    pub inputs: Vec<Arc<LinkedTensor>>,
    pub outputs: Vec<Arc<LinkedTensor>>,
}

impl OpTrait for Operator {
    type TensorPos = WeakTensor;

    #[inline]
    fn op_type(&self) -> &OpType {
        &self.op_type
    }

    #[inline]
    fn inputs(&self) -> Vec<Self::TensorPos> {
        self.inputs
            .iter()
            .map(Arc::downgrade)
            .map(WeakTensor)
            .collect()
    }

    #[inline]
    fn outputs(&self) -> Vec<Self::TensorPos> {
        self.outputs
            .iter()
            .map(Arc::downgrade)
            .map(WeakTensor)
            .collect()
    }
}
