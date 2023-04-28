use super::broadcast;
use crate::Tensor;

pub(super) fn infer(x: &Tensor, y: &Tensor) -> Tensor {
    assert_eq!(x.dtype, y.dtype, "MatMul needs 2 tensor with same type");
    let dtype = x.dtype;
    let x = match x.shape.as_slice() {
        &[] => unreachable!(),
        &[x] => vec![1, x],
        others => Vec::from(others),
    };
    let y = match y.shape.as_slice() {
        &[] => unreachable!(),
        &[y] => vec![y, 1],
        others => Vec::from(others),
    };
    let (xhead, xtail) = x.split_at(x.len() - 2);
    let (yhead, ytail) = y.split_at(y.len() - 2);
    let mut shape = broadcast::multidirection(&[xhead, yhead]).unwrap();
    assert_eq!(xtail[1], ytail[0]);
    shape.extend(&[xtail[0], ytail[1]]);
    Tensor {
        shape,
        dtype,
        data: None,
    }
}
