mod graph;

use super::Operator as OpTrait;
use crate::Tensor;
use basic_operator::OpType;
use std::num::NonZeroUsize;

pub use graph::Unigraph;

pub struct Operator {
    op_idx: OpIdx,
    op_type: OpType,
    inputs: Vec<OutletPos>,
    outputs: Vec<Outlet>,
}

impl OpTrait for Operator {
    type TensorPos = OutletPos;

    #[inline]
    fn op_type(&self) -> &OpType {
        &self.op_type
    }

    #[inline]
    fn inputs(&self) -> Vec<Self::TensorPos> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<Self::TensorPos> {
        (0..self.outputs.len())
            .map(|i| OutletPos {
                op_idx: self.op_idx,
                slot: i,
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct OpIdx(Option<NonZeroUsize>);

impl OpIdx {
    #[inline]
    const fn new_unchecked(idx: usize) -> Self {
        Self(Some(unsafe { NonZeroUsize::new_unchecked(idx + 1) }))
    }

    #[inline]
    fn get(&self) -> Option<usize> {
        self.0.as_ref().map(|x| x.get() - 1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OutletPos {
    op_idx: OpIdx,
    slot: usize,
}

#[allow(unused)]
struct InletPos {
    op_idx: OpIdx,
    slot: usize,
}

struct Outlet {
    targets: Vec<InletPos>,
    tensor: Tensor,
}
