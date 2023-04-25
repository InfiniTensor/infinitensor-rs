use crate::{operator::Conv, Unigraph};
use common::OpType;

pub struct SingleOp;

pub fn partition(mut g: Unigraph) -> Vec<(Unigraph, SingleOp)> {
    core::mem::replace(&mut g.ops, Vec::new())
        .into_iter()
        .map(|op| {
            let mut g = Unigraph::new();
            g.push_op(op.op_type, op.inputs, op.outputs);
            (g, SingleOp)
        })
        .collect()
}

pub fn mutate(g: &Unigraph, _: &SingleOp) -> Vec<Unigraph> {
    let mut ans = Vec::new();
    match g.ops.first().unwrap().op_type {
        OpType::Conv => {
            let conv = Conv::new(g.ops.first().unwrap());
            let i_shape = conv.input().shape();
            let k_shape = conv.kernel().shape();
            let dilations = conv.dilations().data().as_slice::<usize>();
            let strides = conv.strides().data().as_slice::<usize>();
            // assert(conv.input()->data_type == conv.kernel()->data_type);
            let dt = conv.input().data_type();

            if i_shape[1] != k_shape[1] || strides.iter().any(|x| *x != 1) {
            } else if &k_shape[2..=3] == &[1, 1] {
                let mut g = Unigraph::new();

                ans.push(g);
            } else if dilations.iter().any(|x| *x > 1) {
                let mut g = Unigraph::new();

                ans.push(g);
            }
        }
        _ => {}
    };
    ans
}
