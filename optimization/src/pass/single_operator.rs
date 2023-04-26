use std::sync::Arc;

use crate::{operator::Conv, Tensor, Unigraph};
use basic_operator::{infer, OpType};
use common::Data;

#[derive(Debug)]
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
            let [n,c,h,w] = *conv.input().shape() else {
                unreachable!()
            };
            let [f,c_,r,s] = *conv.kernel().shape() else {
                unreachable!()
            };
            let dilations = conv.dilations().data().as_slice::<i32>();
            let strides = conv.strides().data().as_slice::<i32>();
            debug_assert_eq!(conv.input().data_type(), conv.kernel().data_type());
            let dt = conv.input().data_type();

            if c != c_ || strides.iter().any(|x| *x != 1) {
                // nothing to do
            } else if r == 1 && s == 1 {
                let mut mutant = Unigraph::new();

                // (input, "nchw"->"nhwc") -|transpose|-> tranposed -|reshape|-> t0
                let (tranposed, permutation) = transpose(&conv.input(), "nchw", "nhwc");
                mutant.push_op(
                    OpType::Transpose,
                    vec![conv.input().clone(), permutation],
                    vec![tranposed.clone()],
                );
                let t0 = Tensor::share(vec![n * h * w, c], dt, Data::empty());
                mutant.push_op(OpType::Reshape, vec![tranposed], vec![t0.clone()]);

                // (kernel, "fcrs"->"cfrs") -|transpose|-> tranposed -|reshape|-> t1
                let (tranposed, permutation) = transpose(&conv.kernel(), "fcrs", "cfrs");
                mutant.push_op(
                    OpType::Transpose,
                    vec![conv.kernel().clone(), permutation],
                    vec![tranposed.clone()],
                );
                let t1 = Tensor::share(vec![c, f], dt, Data::empty());
                mutant.push_op(OpType::Reshape, vec![tranposed], vec![t1.clone()]);

                // (t0, t1) -|matmul|-> x -|reshape|-> t2
                let x = Tensor::share(infer::matmul(t0.shape(), t1.shape()), dt, Data::empty());
                mutant.push_op(OpType::MatMul, vec![t0, t1], vec![x.clone()]);
                let t2 = Tensor::share(vec![n, h, w, f], dt, Data::empty());
                mutant.push_op(OpType::Reshape, vec![x], vec![t2.clone()]);

                // (t2, "nhwf"->"nfhw") -|transpose|-> output
                let (tranposed, permutation) = transpose(&t2, "nhwf", "nfhw");
                assert_eq!(tranposed.shape(), conv.output().shape());
                mutant.push_op(
                    OpType::Transpose,
                    vec![t2, permutation],
                    vec![conv.output().clone()],
                );

                ans.push(mutant);
            } else if dilations.iter().any(|x| *x > 1) {
                let mut g = Unigraph::new();

                ans.push(g);
            }
        }
        _ => {}
    };
    ans
}

#[inline]
fn transpose(input: &Tensor, src: &str, tgt: &str) -> (Arc<Tensor>, Arc<Tensor>) {
    let (shape, permute) = infer::transpose(input.shape(), src.as_bytes(), tgt.as_bytes());
    (
        Tensor::share(shape, input.data_type(), Data::empty()),
        Tensor::share_vec(permute),
    )
}

#[test]
fn test_1x1_conv() {
    use common::DataType;

    let mut g = Unigraph::new();
    let input = Tensor::share(vec![1, 3, 8, 8], DataType::FLOAT, Data::empty());
    let kernel = Tensor::share(vec![16, 3, 1, 1], DataType::FLOAT, Data::empty());
    let output = Tensor::share(vec![1, 16, 8, 8], DataType::FLOAT, Data::empty());
    g.push_op(
        OpType::Conv,
        vec![
            input,
            kernel,
            Tensor::share_vec(vec![1, 1]),
            Tensor::share_vec(vec![0, 0]),
            Tensor::share_vec(vec![1, 1]),
        ],
        vec![output],
    );

    println!("{g}");
    println!("=========================================================");

    let partition = g.partition(&partition);
    println!("{partition}");
    println!("=========================================================");

    let mutation = partition.mutate(&mutate);
    println!("{mutation}");
    println!("=========================================================");
}
