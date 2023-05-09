use crate::{linked::LinkedTensor, Graph, Operator};
use std::collections::BTreeMap;

impl<Op: Operator> From<&dyn Graph<Op = Op>> for super::Graph {
    fn from(graph: &dyn Graph<Op = Op>) -> Self {
        let mut ans = Self::new();
        let mut tensors = BTreeMap::new();
        for op in graph.ops() {
            let inputs = op
                .inputs()
                .iter()
                .map(|t| {
                    tensors
                        .entry(t.clone())
                        .or_insert_with_key(|key| {
                            let tensor = graph.get_tensor(key);
                            LinkedTensor::share(tensor.shape, tensor.dtype, tensor.data)
                        })
                        .clone()
                })
                .collect();
            let outputs = op
                .outputs()
                .iter()
                .map(|t| {
                    tensors
                        .entry(t.clone())
                        .or_insert_with_key(|key| {
                            let tensor = graph.get_tensor(key);
                            LinkedTensor::share(tensor.shape, tensor.dtype, tensor.data)
                        })
                        .clone()
                })
                .collect();
            ans.push_op(op.op_type().clone(), inputs, outputs);
        }
        ans
    }
}
