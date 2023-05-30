use super::{InletPos, Operator, OperatorPos, Outlet, OutletPos};
use crate::{infer, Tensor};
use basic_operator::OpType;
use common::{AsDataType, Data, DataType};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub struct Graph {
    operator_id: usize,
    operators: BTreeMap<usize, Operator>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            operator_id: 2,
            operators: BTreeMap::from([
                (
                    0,
                    Operator {
                        op_id: 0,
                        op_type: "Input".to_string(),
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                    },
                ),
                (
                    1,
                    Operator {
                        op_id: 1,
                        op_type: "Output".to_string(),
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                    },
                ),
            ]),
        }
    }
}

impl Graph {
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn push_input(&mut self, tensor: Tensor) -> OutletPos {
        let inputs = &mut self.operators.get_mut(&0).unwrap().outputs;
        let slot = inputs.len();
        inputs.push(Outlet {
            targets: Vec::new(),
            tensor,
        });
        OutletPos { op_id: 0, slot }
    }

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

    pub fn set_output(&mut self, outlet: OutletPos) {
        self.operators.get_mut(&1).unwrap().inputs.push(outlet);
    }

    pub fn push_op(&mut self, op_type: String, inputs: Vec<OutletPos>) -> OperatorPos {
        let op_type_ = match op_type.parse::<OpType>() {
            Ok(op_type) => op_type,
            Err(_) => panic!("Unknown operator type: {}", op_type),
        };
        let outputs = infer::infer(
            op_type_,
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
    ) -> OperatorPos {
        self.push_op_inner(
            name,
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
        op_type: String,
        inputs: Vec<OutletPos>,
        outputs: Vec<Tensor>,
    ) -> OperatorPos {
        let op_id = self.operator_id;
        self.operator_id += 1;
        for (slot, pos) in inputs.iter().enumerate() {
            self.get_outlet_mut(pos)
                .targets
                .push(InletPos { op_id, slot });
        }
        let output_len = outputs.len();
        self.operators.entry(op_id).or_insert(Operator {
            op_id,
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
        OperatorPos { op_id, output_len }
    }

    fn get_outlet(&self, pos: &OutletPos) -> &Outlet {
        &self.operators[&pos.op_id].outputs[pos.slot]
    }

    fn get_outlet_mut(&mut self, pos: &OutletPos) -> &mut Outlet {
        self.operators
            .get_mut(&pos.op_id)
            .unwrap()
            .outputs
            .get_mut(pos.slot)
            .unwrap()
    }
}

impl Graph {
    pub fn dce(&mut self) {
        let mut edge = self.operators[&1].inputs.clone();
        let mut operators = self.operators.keys().copied().collect::<BTreeSet<_>>();
        let mut connected = BTreeSet::from([1usize]);
    }

    pub fn remove_operator(&mut self, operator: OperatorPos) {
        todo!()
    }
}
