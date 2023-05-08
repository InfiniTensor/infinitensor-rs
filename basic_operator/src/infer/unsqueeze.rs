use std::collections::BTreeSet;

pub fn infer(data: &[usize], axes: &[i64]) -> Vec<usize> {
    if axes.is_empty() {
        data.iter().copied().collect()
    } else {
        let len = (data.len() + axes.len()) as i64;
        let axes = axes
            .iter()
            .copied()
            .map(|x| if x < 0 { len + x } else { x } as usize)
            .collect::<BTreeSet<_>>();

        let len = len as usize;
        debug_assert_eq!(data.len() + axes.len(), len);
        let mut ans = Vec::with_capacity(len);
        let mut iter = data.iter();
        for i in 0..len as usize {
            ans.push(if axes.contains(&i) {
                1
            } else {
                *iter.next().unwrap()
            });
        }
        ans
    }
}

#[test]
fn test_rehape() {
    assert_eq!(
        infer(&[1, 2, 1, 2, 3, 1, 2, 3, 4], &[0, -4]),
        &[2, 1, 2, 3, 2, 3, 4]
    );
    assert_eq!(
        infer(&[1, 2, 1, 2, 3, 1, 2, 3, 4], &[]),
        &[2, 2, 3, 2, 3, 4]
    );
}
