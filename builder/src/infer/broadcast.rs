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

pub(super) fn unidirection(target: &[usize], source: &[usize]) -> Option<Vec<usize>> {
    let mut ans = vec![0; target.len()];
    let mut iter = source.iter().rev();
    for (y, x) in ans.iter_mut().rev().zip(target.iter().rev().copied()) {
        *y = if [1, x].contains(iter.next().unwrap_or(&1)) {
            x
        } else {
            return None;
        }
    }
    Some(ans)
}
