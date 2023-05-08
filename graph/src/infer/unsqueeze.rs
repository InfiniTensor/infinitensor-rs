use crate::Tensor;
use basic_operator::infer;
use common::DataType;

pub(super) fn infer(data: &Tensor, axes: &Tensor) -> Tensor {
    assert!(
        matches!(axes.dtype, DataType::INT64),
        "Axes data type must be int64"
    );

    Tensor {
        shape: infer::unsqueeze(
            &data.shape,
            axes.data.as_ref().unwrap().as_typed_slice::<i64>(),
        ),
        dtype: data.dtype,
        data: data.data.clone(),
    }
}
