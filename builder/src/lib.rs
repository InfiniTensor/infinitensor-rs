mod infer;

use basic_operator::OpType;
use common::{AsDataType, Data, DataType};
use std::{num::NonZeroUsize, sync::Arc};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct OpIdx(Option<NonZeroUsize>);

impl OpIdx {
    #[inline]
    pub const fn new_unchecked(idx: usize) -> Self {
        Self(Some(unsafe { NonZeroUsize::new_unchecked(idx + 1) }))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct InletPos {
    op_idx: OpIdx,
    slot: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OutletPos {
    op_idx: OpIdx,
    slot: usize,
}

struct Outlet {
    targets: Vec<InletPos>,
    tensor: Tensor,
}

#[derive(Clone)]
struct Tensor {
    shape: Vec<usize>,
    dtype: DataType,
    data: Option<Arc<Data>>,
}

impl Tensor {
    #[inline]
    pub fn clone_info(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dtype: self.dtype,
            data: None,
        }
    }
}

struct Operator {
    op_type: OpType,
    inputs: Vec<OutletPos>,
    outputs: Vec<Outlet>,
}

pub struct Builder {
    inputs: Vec<Outlet>,
    outputs: Vec<InletPos>,
    operators: Vec<Operator>,
}

impl Default for Builder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    #[inline]
    pub const fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            operators: Vec::new(),
        }
    }

    pub fn push_input<T: AsDataType>(
        &mut self,
        shape: Vec<usize>,
        data: Option<Vec<T>>,
    ) -> OutletPos {
        let slot = self.inputs.len();
        self.inputs.push(Outlet {
            targets: Vec::new(),
            tensor: Tensor {
                shape,
                dtype: T::as_data_type(),
                data: data.map(Data::cpu).map(Arc::new),
            },
        });
        OutletPos {
            op_idx: OpIdx(None),
            slot,
        }
    }

    pub fn push_operator(&mut self, op_type: OpType, inputs: Vec<OutletPos>) -> Vec<OutletPos> {
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

    pub fn push_custom<I>(
        &mut self,
        name: String,
        inputs: Vec<OutletPos>,
        outputs: I,
    ) -> Vec<OutletPos>
    where
        I: IntoIterator<Item = (Vec<usize>, DataType)>,
    {
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
        let outlet_len = outputs.len();
        for (slot, pos) in inputs.iter().enumerate() {
            self.get_outlet_mut(pos).targets.push(InletPos {
                op_idx: OpIdx::new_unchecked(op_idx),
                slot,
            });
        }
        self.operators.push(Operator {
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
        (0..outlet_len)
            .map(|slot| OutletPos {
                op_idx: OpIdx::new_unchecked(op_idx),
                slot,
            })
            .collect()
    }

    fn get_outlet(&self, pos: &OutletPos) -> &Outlet {
        if let Some(node) = pos.op_idx.0 {
            &self.operators[node.get() - 1].outputs[pos.slot]
        } else {
            &self.inputs[pos.slot]
        }
    }

    fn get_outlet_mut(&mut self, pos: &OutletPos) -> &mut Outlet {
        if let Some(node) = pos.op_idx.0 {
            &mut self.operators[node.get() - 1].outputs[pos.slot]
        } else {
            &mut self.inputs[pos.slot]
        }
    }
}
