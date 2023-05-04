use super::{InletPos, OpIdx, Operator, Outlet, OutletPos};
use crate::{
    graph::{Graph, Operator as OpTrait},
    infer, Tensor,
};
use basic_operator::OpType;
use common::{AsDataType, Data, DataType};
use std::{fmt, sync::Arc};

pub struct Unigraph {
    inputs: Vec<Outlet>,
    outputs: Vec<InletPos>,
    operators: Vec<Operator>,
}

impl Unigraph {
    #[inline]
    pub const fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            operators: Vec::new(),
        }
    }

    pub fn push_input(&mut self, tensor: Tensor) -> OutletPos {
        let slot = self.inputs.len();
        self.inputs.push(Outlet {
            targets: Vec::new(),
            tensor,
        });
        OutletPos {
            op_idx: OpIdx(None),
            slot,
        }
    }

    #[inline]
    pub fn push_typed_input<T: AsDataType>(
        &mut self,
        shape: Vec<usize>,
        data: Option<Vec<T>>,
    ) -> OutletPos {
        self.push_input(Tensor {
            shape,
            dtype: T::as_data_type(),
            data: data.map(Data::cpu).map(Arc::new),
        })
    }

    #[inline]
    pub fn set_output(&mut self, OutletPos { op_idx, slot }: OutletPos) {
        self.outputs.push(InletPos { op_idx, slot });
    }

    pub fn push_op(&mut self, op_type: OpType, inputs: Vec<OutletPos>) -> Vec<OutletPos> {
        if let OpType::Custom(_) = op_type {
            panic!("Use `push_custom` instead to push custom operator.")
        }
        let outputs = infer::infer(
            op_type.clone(),
            &inputs
                .iter()
                .map(|x| &self.get_outlet(x).tensor)
                .collect::<Vec<_>>(),
        );
        self.push_op_inner(op_type, inputs, outputs)
    }

    pub fn push_custom(
        &mut self,
        name: String,
        inputs: Vec<OutletPos>,
        outputs: impl IntoIterator<Item = (Vec<usize>, DataType)>,
    ) -> Vec<OutletPos> {
        self.push_op_inner(
            OpType::Custom(name),
            inputs,
            outputs
                .into_iter()
                .map(|(shape, dtype)| Tensor {
                    shape,
                    dtype,
                    data: None,
                })
                .collect(),
        )
    }

    fn push_op_inner(
        &mut self,
        op_type: OpType,
        inputs: Vec<OutletPos>,
        outputs: Vec<Tensor>,
    ) -> Vec<OutletPos> {
        let op_idx = self.operators.len();
        for (slot, pos) in inputs.iter().enumerate() {
            self.get_outlet_mut(pos).targets.push(InletPos {
                op_idx: OpIdx::new_unchecked(op_idx),
                slot,
            });
        }
        self.operators.push(Operator {
            op_idx: OpIdx::new_unchecked(op_idx),
            op_type,
            inputs,
            outputs: outputs
                .into_iter()
                .map(|t| Outlet {
                    targets: Vec::new(),
                    tensor: t,
                })
                .collect(),
        });
        self.operators.last().unwrap().outputs()
    }

    fn get_outlet(&self, pos: &OutletPos) -> &Outlet {
        if let Some(i) = pos.op_idx.get() {
            &self.operators[i].outputs[pos.slot]
        } else {
            &self.inputs[pos.slot]
        }
    }

    fn get_outlet_mut(&mut self, pos: &OutletPos) -> &mut Outlet {
        if let Some(i) = pos.op_idx.get() {
            &mut self.operators[i - 1].outputs[pos.slot]
        } else {
            &mut self.inputs[pos.slot]
        }
    }
}

impl Default for Unigraph {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Unigraph {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_impl(f)
    }
}

impl Graph for Unigraph {
    type Op = Operator;

    #[inline]
    fn ops(&self) -> &[Self::Op] {
        &self.operators
    }

    #[inline]
    fn get_tensor(&self, pos: &<Self::Op as OpTrait>::TensorPos) -> Tensor {
        self.get_outlet(pos).tensor.clone()
    }
}
