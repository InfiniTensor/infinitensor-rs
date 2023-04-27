use super::broadcast;
use crate::Tensor;
use common::DataType;

pub(super) fn infer(x: &Tensor, y: &Tensor) -> Tensor {
    if x.dtype != y.dtype {
        panic!("Compair need 2 tensor with same type")
    } else if !x.dtype.is_numeric() {
        panic!("Compair only support numeric type")
    } else {
        Tensor {
            shape: broadcast::multidirection(&[&x.shape, &y.shape]).expect("Failed to broadcast"),
            dtype: DataType::BOOL,
            data: None,
        }
    }
}
