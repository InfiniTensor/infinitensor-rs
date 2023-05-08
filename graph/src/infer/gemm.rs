use crate::{infer::broadcast, Tensor};

pub(super) fn infer(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    trans_a: &Tensor,
    trans_b: &Tensor,
) -> Tensor {
    let dtype = a.dtype;
    assert_eq!(dtype, b.dtype);
    assert_eq!(dtype, c.dtype);

    assert!(alpha.dtype.is_float());
    assert!(alpha.is_scalar());
    assert!(beta.dtype.is_float());
    assert!(beta.is_scalar());
    assert!(trans_a.is_typed_scalar::<bool>());
    assert!(trans_b.is_typed_scalar::<bool>());

    let trans_a = trans_a
        .data
        .as_ref()
        .expect("TransA must be const")
        .as_typed_slice::<bool>()[0];
    let trans_b = trans_b
        .data
        .as_ref()
        .expect("TransA must be const")
        .as_typed_slice::<bool>()[0];

    let a = match a.shape.as_slice() {
        &[x, y] => {
            if trans_a {
                [x, y]
            } else {
                [y, x]
            }
        }
        _ => panic!(),
    };
    let b = match b.shape.as_slice() {
        &[x, y] => {
            if trans_b {
                [x, y]
            } else {
                [y, x]
            }
        }
        _ => panic!(),
    };
    assert_eq!(a[1], b[0]);
    Tensor {
        shape: broadcast::unidirection(&[a[0], b[1]], &c.shape).expect("Broudcast failed"),
        dtype,
        data: None,
    }
}
