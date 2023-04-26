use crate::{operator::Conv, Tensor, Unigraph};
use common::{Data, OpType};
use std::collections::BTreeMap;

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
            let i_shape = conv.input().shape();
            let k_shape = conv.kernel().shape();
            let dilations = conv.dilations().data().as_slice::<i32>();
            let strides = conv.strides().data().as_slice::<i32>();
            // assert(conv.input()->data_type == conv.kernel()->data_type);
            let dt = conv.input().data_type();

            if i_shape[1] != k_shape[1] || strides.iter().any(|x| *x != 1) {
                // nothing to do
            } else if &k_shape[2..=3] == &[1, 1] {
                let mut mutant = Unigraph::new();

                // (input, "nchw"->"nhwc") -|transpose|-> tranposed -|reshape|-> t0
                let (shape, permute) = transpose(i_shape, "nchw", "nhwc");
                let tranposed = Tensor::share(shape, dt, Data::empty());
                let permutation = Tensor::share_vec(permute);
                mutant.push_op(
                    OpType::Transpose,
                    vec![conv.input().clone(), permutation],
                    vec![tranposed.clone()],
                );
                let t0 = Tensor::share(
                    reshape(tranposed.shape(), &[Rule::Mul(3), Rule::Keep]),
                    dt,
                    Data::empty(),
                );
                mutant.push_op(OpType::Reshape, vec![tranposed], vec![t0.clone()]);

                // (kernel,"fcrs"->"cfrs") -|transpose|-> tranposed -|reshape|-> t1
                let (shape, permute) = transpose(k_shape, "fcrs", "cfrs");
                let tranposed = Tensor::share(shape, dt, Data::empty());
                let permutation = Tensor::share_vec(permute);
                mutant.push_op(
                    OpType::Transpose,
                    vec![conv.kernel().clone(), permutation],
                    vec![tranposed.clone()],
                );
                let t1 = Tensor::share(
                    reshape(tranposed.shape(), &[Rule::Keep, Rule::Mul(3)]),
                    dt,
                    Data::empty(),
                );
                mutant.push_op(OpType::Reshape, vec![tranposed], vec![t1.clone()]);

                // (t0,t1) -|matmul|-> x -|reshape|-> t2
                let x = Tensor::share(vec![t0.shape()[0], t1.shape()[1]], dt, Data::empty());
                mutant.push_op(OpType::MatMul, vec![t0, t1], vec![x.clone()]);
                let t2 = Tensor::share(
                    vec![i_shape[0], i_shape[2], i_shape[3], k_shape[0]],
                    dt,
                    Data::empty(),
                );
                mutant.push_op(OpType::Reshape, vec![x], vec![t2.clone()]);

                // (t2,"nhwf"->"nfhw") -|transpose|-> output
                let (shape, permute) = transpose(t2.shape(), "nhwf", "nfhw");
                assert_eq!(shape, conv.output().shape());
                let permutation = Tensor::share_vec(permute);
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

// 1st: new shape
// 2nd: permutation
fn transpose(shape: &[usize], src: &str, tgt: &str) -> (Vec<usize>, Vec<i32>) {
    let mut indices = BTreeMap::new();
    for (i, c) in src.chars().enumerate() {
        indices.insert(c, i);
    }
    let mut ans = (vec![0; shape.len()], vec![0; shape.len()]);
    for (i, c) in tgt.chars().enumerate() {
        let idx = indices[&c];
        ans.0[i] = shape[idx];
        ans.1[i] = idx as i32;
    }
    ans
}

#[test]
fn test_transpose() {
    assert_eq!(
        transpose(&[1, 2, 3, 4], "nchw", "nhwc"),
        (vec![1, 3, 4, 2], vec![0, 2, 3, 1])
    );
}

enum Rule {
    Keep,
    Mul(usize),
    Foreward(usize),
    Backward(usize),
}

fn reshape(shape: &[usize], rule: &[Rule]) -> Vec<usize> {
    let mut shape = shape.into_iter();
    let mut ans = Vec::new();
    for r in rule {
        match r {
            Rule::Keep => {
                ans.push(*shape.next().unwrap());
            }
            Rule::Mul(n) => {
                ans.push((0..*n).map(|_| shape.next().unwrap()).product());
            }
            Rule::Foreward(x) => {
                ans.push(*x);
                ans.push(shape.next().unwrap() / x);
            }
            Rule::Backward(x) => {
                ans.push(shape.next().unwrap() / x);
                ans.push(*x);
            }
        }
    }
    // assert!(shape.next().is_none());
    ans
}

#[test]
fn test_reshape() {
    assert_eq!(
        reshape(
            &[1, 2, 3, 4, 5, 6, 7],
            &[
                Rule::Keep,
                Rule::Mul(3),
                Rule::Keep,
                Rule::Foreward(2),
                Rule::Backward(1),
            ]
        ),
        &[1, 24, 5, 2, 3, 7, 1]
    );
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
