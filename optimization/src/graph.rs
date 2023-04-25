use crate::{Tensor, TensorPos};
use common::OpType;
use std::sync::{
    atomic::{AtomicUsize, Ordering::AcqRel},
    Arc,
};

pub struct Operator {
    pub(crate) op_type: OpType,
    pub(crate) inputs: Vec<Arc<Tensor>>,
    pub(crate) outputs: Vec<Arc<Tensor>>,
}

pub struct OpRef {
    graph: usize,
    op: usize,
}

pub struct Unigraph {
    id: usize,
    pub(crate) ops: Vec<Operator>,
}

impl Unigraph {
    #[inline]
    pub fn new() -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ID.fetch_add(1, AcqRel),
            ops: Default::default(),
        }
    }

    pub fn push_op(
        &mut self,
        op_type: OpType,
        inputs: Vec<Arc<Tensor>>,
        outputs: Vec<Arc<Tensor>>,
    ) -> OpRef {
        use std::collections::hash_map::Entry::*;

        let ans = OpRef {
            graph: self.id,
            op: self.ops.len(),
        };

        for (i, input) in inputs.iter().enumerate() {
            match input.target.lock().unwrap().entry(ans.graph) {
                Occupied(mut entry) => {
                    entry.get_mut().push(TensorPos { op: ans.op, idx: i });
                }
                Vacant(entry) => {
                    entry.insert(vec![TensorPos { op: ans.op, idx: i }]);
                }
            }
        }
        for (i, output) in outputs.iter().enumerate() {
            match output.source.lock().unwrap().entry(ans.graph) {
                Occupied(_) => panic!("Tensor source exist"),
                Vacant(entry) => {
                    entry.insert(TensorPos { op: ans.op, idx: i });
                }
            }
        }

        self.ops.push(Operator {
            op_type,
            inputs,
            outputs,
        });
        ans
    }
}

impl Drop for Unigraph {
    fn drop(&mut self) {
        for op in &self.ops {
            for input in &op.inputs {
                input.target.lock().unwrap().remove(&self.id);
            }
            for output in &op.outputs {
                output.source.lock().unwrap().remove(&self.id);
            }
        }
    }
}
