use crate::Tensor;
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
    assert_eq!(dim, data.shape.len());

    let mut void = None;
    let mut ans = shape
        .iter()
        .zip(data.shape.iter())
        .enumerate()
        .map(|(i, (reshape, origin))| match *reshape {
            -1 => match void.replace(i) {
                Some(_) => panic!("Only one dimension can be inferred"),
                None => 1,
            },
            0 => *origin,
            x if x < 0 => unreachable!(),
            x => x as usize,
        })
        .collect::<Vec<_>>();
    if let Some(void) = void {
        ans[void] = data.shape.iter().product::<usize>() / ans.iter().product::<usize>();
    }

    Tensor {
        shape: ans,
        dtype: data.dtype,
        data: data.data.clone(),
    }
}
