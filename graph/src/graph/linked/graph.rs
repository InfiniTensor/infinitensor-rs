use super::tensor::{LinkedTensor, TensorPos};
use basic_operator::OpType;
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering::AcqRel},
        Arc,
    },
};

pub struct Operator {
    pub op_type: OpType,
    pub inputs: Vec<Arc<LinkedTensor>>,
    pub outputs: Vec<Arc<LinkedTensor>>,
}

pub struct Unigraph {
    id: usize,
    ops: Vec<Operator>,
}

impl Default for Unigraph {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
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

    #[inline]
    pub fn ops(&self) -> &[Operator] {
        &self.ops
    }

    fn drop_inplace(&self) {
        for op in &self.ops {
            for input in &op.inputs {
                input.target.lock().unwrap().remove(&self.id);
            }
            for output in &op.outputs {
                output.source.lock().unwrap().remove(&self.id);
            }
        }
    }

    #[inline]
    pub fn take_ops(mut self) -> Vec<Operator> {
        self.drop_inplace();
        core::mem::take(&mut self.ops)
    }

    pub fn push_op(
        &mut self,
        op_type: OpType,
        inputs: Vec<Arc<LinkedTensor>>,
        outputs: Vec<Arc<LinkedTensor>>,
    ) -> &Operator {
        use std::collections::hash_map::Entry::*;

        let op = self.ops.len();

        for (idx, input) in inputs.iter().enumerate() {
            match input.target.lock().unwrap().entry(self.id) {
                Occupied(mut entry) => {
                    entry.get_mut().push(TensorPos { op, idx });
                }
                Vacant(entry) => {
                    entry.insert(vec![TensorPos { op, idx }]);
                }
            }
        }
        for (idx, output) in outputs.iter().enumerate() {
            match output.source.lock().unwrap().entry(self.id) {
                Occupied(_) => panic!("Tensor source exist"),
                Vacant(entry) => {
                    entry.insert(TensorPos { op, idx });
                }
            }
        }

        self.ops.push(Operator {
            op_type,
            inputs,
            outputs,
        });

        self.ops.last().unwrap()
    }
}

impl Drop for Unigraph {
    #[inline]
    fn drop(&mut self) {
        self.drop_inplace();
    }
}

impl fmt::Display for Unigraph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Unigraph {}", self.id)?;
        writeln!(f, "------------------------")?;

        let mut id = 0;
        let mut tensors = BTreeMap::new();
        for op in &self.ops {
            use std::collections::btree_map::Entry::*;

            writeln!(f)?;
            let origin = id;
            for t in op.inputs.iter().chain(&op.outputs) {
                if let Vacant(entry) = tensors.entry(Arc::as_ptr(t)) {
                    writeln!(f, "_{} = {t}", id)?;
                    entry.insert(id);
                    id += 1;
                }
            }
            if id != origin {
                writeln!(f)?;
            }
            writeln!(
                f,
                "({}) = {:?}({})",
                op.outputs
                    .iter()
                    .map(|t| format!("_{}", tensors[&Arc::as_ptr(t)]))
                    .collect::<Vec<_>>()
                    .join(", "),
                op.op_type,
                op.inputs
                    .iter()
                    .map(|t| format!("_{}", tensors[&Arc::as_ptr(t)]))
                    .collect::<Vec<_>>()
                    .join(", "),
            )?;
        }

        Ok(())
    }
}
