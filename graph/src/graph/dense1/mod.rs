mod graph;

use crate::Tensor;
use core::cmp::Ordering;
use std::ops::Index;

pub use graph::Graph;

pub struct Operator {
    op_id: usize,
    op_type: String,
    inputs: Vec<OutletPos>,
    outputs: Vec<Outlet>,
}

pub struct OperatorPos {
    op_id: usize,
    output_len: usize,
}

impl OperatorPos {
    pub fn get(&self, slot: usize) -> OutletPos {
        OutletPos {
            op_id: self.op_id,
            slot,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OutletPos {
    op_id: usize,
    slot: usize,
}

impl PartialOrd for OutletPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.op_id.partial_cmp(&other.op_id) {
            Some(Ordering::Equal) => self.slot.partial_cmp(&other.slot),
            ord => ord,
        }
    }
}

impl Ord for OutletPos {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.op_id.cmp(&other.op_id) {
            Ordering::Equal => self.slot.cmp(&other.slot),
            ord => ord,
        }
    }
}

#[allow(unused)]
struct InletPos {
    op_id: usize,
    slot: usize,
}

struct Outlet {
    targets: Vec<InletPos>,
    tensor: Tensor,
}
