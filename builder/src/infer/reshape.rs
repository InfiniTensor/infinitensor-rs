use crate::Tensor;
use basic_operator::infer;
use common::DataType;

pub(super) fn infer(data: &Tensor, shape: &Tensor) -> Tensor {
    assert!(
        matches!(shape.dtype, DataType::INT64),
        "Shape data type must be int64"
    );

    let &[dim] = shape.shape.as_slice()
    else { panic!("Shape must be a vectoer") };

    let Some(val) = shape.data.as_ref()
    else { panic!("Shape must be const") };

    let shape = val.as_slice::<i64>();
    assert_eq!(dim, shape.len());

    Tensor {
        shape: infer::reshape(&data.shape, shape),
        dtype: data.dtype,
        data: data.data.clone(),
    }
}
