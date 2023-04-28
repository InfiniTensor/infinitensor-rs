use super::broadcast;
use crate::Tensor;
use basic_operator::Binary;

pub(super) fn infer(ty: Binary, x: &Tensor, y: &Tensor) -> Tensor {
    use Binary::*;
    match ty {
        Add | Sub | Mul | Div | Mod => {
            if x.dtype != y.dtype {
                panic!("Binary need 2 tensor with same type")
            } else if !x.dtype.is_numeric() {
                panic!("Binary only support numeric type")
            } else {
                Tensor {
                    shape: broadcast::multidirection(&[&x.shape, &y.shape])
                        .expect("Failed to broadcast"),
                    dtype: x.dtype,
                    data: None,
                }
            }
        }
        And | Or | Xor => {
            if x.dtype != y.dtype {
                panic!("Binary need 2 tensor with same type")
            } else if !x.dtype.is_bool() {
                panic!("Binary only support bool type")
            } else {
                Tensor {
                    shape: broadcast::multidirection(&[&x.shape, &y.shape])
                        .expect("Failed to broadcast"),
                    dtype: x.dtype,
                    data: None,
                }
            }
        }
        Pow => {
            if !x.dtype.is_float() || !y.dtype.is_numeric() {
                panic!("Binary arguments type must be (float, numeric)")
            } else {
                Tensor {
                    shape: broadcast::multidirection(&[&x.shape, &y.shape])
                        .expect("Failed to broadcast"),
                    dtype: x.dtype,
                    data: None,
                }
            }
        }
        BitShift | BitwiseAnd | BitwiseNot | BitwiseOr | BitwiseXor => {
            if x.dtype != y.dtype {
                panic!("Binary need 2 tensor with same type")
            } else if !x.dtype.is_bits() {
                panic!("Binary only support bits type")
            } else if x.shape != y.shape {
                panic!("Binary need 2 tensor with same shape")
            } else {
                x.clone_info()
            }
        }
    }
}
