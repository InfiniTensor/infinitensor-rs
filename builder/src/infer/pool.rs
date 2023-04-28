use crate::Tensor;
use basic_operator::infer;
use common::DataType;

pub(super) fn infer(
    data: &Tensor,
    kernel: &Tensor,
    dilations: &Tensor,
    pads: &Tensor,
    strides: &Tensor,
) -> Tensor {
    assert!(matches!(dilations.dtype, DataType::INT64));
    assert!(matches!(pads.dtype, DataType::INT64));
    assert!(matches!(strides.dtype, DataType::INT64));

    let kernel = kernel.data.as_ref().expect("Kernel must be const");
    let dilations = dilations.data.as_ref().expect("Dilations must be const");
    let pads = pads.data.as_ref().expect("Pads must be const");
    let strides = strides.data.as_ref().expect("Strides must be const");

    let shape = infer::pool(
        data.shape
            .iter()
            .map(|x| *x as i64)
            .collect::<Vec<_>>()
            .as_slice(),
        kernel.as_slice::<i64>(),
        dilations.as_slice::<i64>(),
        pads.as_slice::<i64>(),
        strides.as_slice::<i64>(),
    );

    Tensor {
        shape: shape.into_iter().map(|x| x as _).collect(),
        dtype: data.dtype,
        data: data.data.clone(),
    }
}
