pub mod dense;
pub mod linked;

use crate::Tensor;
use basic_operator::OpType;
use std::{
    collections::BTreeMap,
    fmt::{self, Display},
};

pub trait Operator {
    type TensorPos: Ord + Clone;

    fn op_type(&self) -> &OpType;
    fn inputs(&self) -> Vec<Self::TensorPos>;
    fn outputs(&self) -> Vec<Self::TensorPos>;
}

pub trait Graph: Display {
    type Op: Operator;

    fn ops(&self) -> &[Self::Op];
    fn get_tensor(&self, pos: &<Self::Op as Operator>::TensorPos) -> Tensor;

    fn fmt_impl(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut id = 0;
        let mut tensors = BTreeMap::new();
        for op in self.ops() {
            use std::collections::btree_map::Entry::*;

            writeln!(f)?;
            let origin = id;
            for t in op.inputs().iter().chain(&op.outputs()) {
                if let Vacant(entry) = tensors.entry(t.clone()) {
                    writeln!(f, "_{id} = {}", self.get_tensor(t))?;
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
                op.outputs()
                    .iter()
                    .map(|t| format!("_{}", tensors[t]))
                    .collect::<Vec<_>>()
                    .join(", "),
                op.op_type(),
                op.inputs()
                    .iter()
                    .map(|t| format!("_{}", tensors[t]))
                    .collect::<Vec<_>>()
                    .join(", "),
            )?;
        }

        Ok(())
    }
}
