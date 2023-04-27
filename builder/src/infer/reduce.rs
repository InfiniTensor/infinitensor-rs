use super::broadcast;
use crate::Tensor;
use basic_operator::Reduce;
use std::collections::BTreeSet;

pub(super) fn infer(ty: Reduce, input: &[&Tensor]) -> Tensor {
    use Reduce::*;
    let &[dtype] = input
        .iter()
        .map(|t| t.dtype)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .as_slice()
        else { panic!("Reduce only support same type") };
    match ty {
        Max | Min if !dtype.is_numeric() => panic!("Reduce only support numeric type"),
        Mean | Sum if !dtype.is_float() => panic!("Reduce only support float type"),
        _ => Tensor {
            shape: broadcast::multidirection(
                input
                    .iter()
                    .map(|t| t.shape.as_slice())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .expect("Failed to broadcast"),
            dtype,
            data: None,
        },
    }
}
