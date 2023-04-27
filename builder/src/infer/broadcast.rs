use std::collections::BTreeSet;

/// See <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md>.
pub(super) fn multidirection(tensors: &[&[usize]]) -> Option<Vec<usize>> {
    let mut ans = vec![0; tensors.iter().map(|s| s.len()).max().unwrap()];
    let mut iters = tensors.iter().map(|s| s.iter().rev()).collect::<Vec<_>>();
    for y in ans.iter_mut().rev() {
        *y = match iters
            .iter_mut()
            .map(|it| it.next().copied().unwrap_or(1))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .as_slice()
        {
            &[] => unreachable!(),
            &[x] | &[1, x] | &[x, 1] => x,
            _ => return None,
        };
    }
    Some(ans)
}
