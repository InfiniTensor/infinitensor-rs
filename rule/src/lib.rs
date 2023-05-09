use basic_operator::OpType;
use graph::Tensor;
use std::{fmt, num::NonZeroUsize};

pub struct Operator {
    op_type: OpType,
    inputs: Vec<OutletPos>,
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

pub struct TargetGraph {
    inputs: Vec<Option<Tensor>>,
    outputs: Vec<OutletPos>,
    operators: Vec<Operator>,
}

pub struct OutletBuilder(OpIdx);

impl OutletBuilder {
    #[inline]
    pub fn slot(&self, slot: usize) -> OutletPos {
        OutletPos {
            op_idx: self.0,
            slot,
        }
    }
}

impl TargetGraph {
    pub fn push_input(&mut self) -> OutletPos {
        let idx = self.inputs.len();
        self.inputs.push(None);
        OutletPos {
            op_idx: OpIdx(None),
            slot: idx,
        }
    }

    pub fn push_data(&mut self, tensor: Tensor) -> OutletPos {
        let idx = self.inputs.len();
        self.inputs.push(Some(tensor));
        OutletPos {
            op_idx: OpIdx(None),
            slot: idx,
        }
    }

    pub fn push_operator(&mut self, op_type: OpType, inputs: Vec<OutletPos>) -> OutletBuilder {
        let idx = self.operators.len();
        self.operators.push(Operator { op_type, inputs });
        OutletBuilder(OpIdx::new_unchecked(idx))
    }

    pub fn set_output(&mut self, OutletPos { op_idx, slot }: OutletPos) {
        self.outputs.push(OutletPos { op_idx, slot });
    }
}

impl fmt::Display for TargetGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Target [{}] -> [{}]",
            self.inputs.len(),
            self.outputs.len()
        )?;
        Ok(())
    }
}

#[test]
fn test() {
    let mut graph = TargetGraph {
        inputs: vec![],
        outputs: vec![],
        operators: vec![],
    };

    let data = graph.push_input();
    let transpose = graph.push_operator(OpType::Transpose, vec![data]).slot(0);
    let reshape = graph
        .push_operator(OpType::Reshape, vec![transpose])
        .slot(0);

    let kernel = graph.push_input();
    let transpose = graph.push_operator(OpType::Transpose, vec![kernel]).slot(0);
}
