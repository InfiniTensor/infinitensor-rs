use crate::operator::Conv;
use basic_operator::{infer, OpType};
use graph::{
    linked::{LinkedTensor, Unigraph},
    Operator,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct SingleOp;

pub fn partition(g: Unigraph) -> Vec<(Unigraph, SingleOp)> {
    g.take_ops()
        .into_iter()
        .map(|op| {
            let mut g = Unigraph::new();
            g.push_op(op.op_type().clone(), op.inputs, op.outputs);
            (g, SingleOp)
        })
        .collect()
}

pub fn mutate(g: &Unigraph, _: &SingleOp) -> Vec<Unigraph> {
    let mut ans = Vec::new();
    let op = g.ops().first().unwrap();
    if let OpType::Conv = op.op_type() {
        let conv = Conv::new(op);
        let [n,c,h,w] = *conv.input().shape() else {
            unreachable!()
        };
        let [f,c_,r,s] = *conv.kernel().shape() else {
            unreachable!()
        };
        let dilations = conv
            .dilations()
            .data()
            .as_ref()
            .unwrap()
            .as_typed_slice::<i32>();
        let strides = conv
            .strides()
            .data()
            .as_ref()
            .unwrap()
            .as_typed_slice::<i32>();
        debug_assert_eq!(conv.input().dtype(), conv.kernel().dtype());
        let dt = conv.input().dtype();

        if c != c_ || strides.iter().any(|x| *x != 1) {
            // nothing to do
        } else if r == 1 && s == 1 {
            let mut mutant = Unigraph::new();

            // (input, "nchw"->"nhwc") -|transpose|-> tranposed -|reshape|-> t0
            let (tranposed, permutation) = transpose(conv.input(), "nchw", "nhwc");
            mutant.push_op(
                OpType::Transpose,
                vec![conv.input().clone(), permutation],
                vec![tranposed.clone()],
            );
            let t0 = LinkedTensor::share(vec![n * h * w, c], dt, None);
            mutant.push_op(OpType::Reshape, vec![tranposed], vec![t0.clone()]);

            // (kernel, "fcrs"->"cfrs") -|transpose|-> tranposed -|reshape|-> t1
            let (tranposed, permutation) = transpose(conv.kernel(), "fcrs", "cfrs");
            mutant.push_op(
                OpType::Transpose,
                vec![conv.kernel().clone(), permutation],
                vec![tranposed.clone()],
            );
            let t1 = LinkedTensor::share(vec![c, f], dt, None);
            mutant.push_op(OpType::Reshape, vec![tranposed], vec![t1.clone()]);

            // (t0, t1) -|matmul|-> x -|reshape|-> t2
            let x = LinkedTensor::share(infer::matmul(t0.shape(), t1.shape()), dt, None);
            mutant.push_op(OpType::MatMul, vec![t0, t1], vec![x.clone()]);
            let t2 = LinkedTensor::share(vec![n, h, w, f], dt, None);
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
            // let mut g = Unigraph::new();

            // ans.push(g);
        }
    };
    ans
}

#[inline]
fn transpose(input: &LinkedTensor, src: &str, tgt: &str) -> (Arc<LinkedTensor>, Arc<LinkedTensor>) {
    let (shape, permute) = infer::transpose(input.shape(), src.as_bytes(), tgt.as_bytes());
    (
        LinkedTensor::share(shape, input.dtype(), None),
        LinkedTensor::share_vec(permute),
    )
}

#[test]
fn test_1x1_conv() {
    use crate::mutation::Partition;
    use common::DataType;

    let mut g = Unigraph::new();
    let input = LinkedTensor::share(vec![1, 3, 8, 8], DataType::FLOAT, None);
    let kernel = LinkedTensor::share(vec![16, 3, 1, 1], DataType::FLOAT, None);
    let output = LinkedTensor::share(vec![1, 16, 8, 8], DataType::FLOAT, None);
    g.push_op(
        OpType::Conv,
        vec![
            input,
            kernel,
            LinkedTensor::share_vec(vec![1, 1]),
            LinkedTensor::share_vec(vec![0, 0]),
            LinkedTensor::share_vec(vec![1, 1]),
        ],
        vec![output],
    );

    println!("{g}");
    println!("=========================================================");

    let partition = Partition::new(g, &partition);
    println!("{partition}");
    println!("=========================================================");

    let mutation = partition.mutate(&mutate);
    println!("{mutation}");
    println!("=========================================================");
}
