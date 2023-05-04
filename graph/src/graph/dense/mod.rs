mod convert;
mod graph;

use super::Operator as OpTrait;
use crate::Tensor;
use basic_operator::OpType;
use core::cmp::Ordering;
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

impl PartialOrd for OutletPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.op_idx.get().partial_cmp(&other.op_idx.get()) {
            Some(Ordering::Equal) => self.slot.partial_cmp(&other.slot),
            ord => ord,
        }
    }
}

impl Ord for OutletPos {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.op_idx.get().cmp(&other.op_idx.get()) {
            Ordering::Equal => self.slot.cmp(&other.slot),
            ord => ord,
        }
    }
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
