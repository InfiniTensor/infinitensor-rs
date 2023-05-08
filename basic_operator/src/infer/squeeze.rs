use std::collections::BTreeSet;

pub fn infer(data: &[usize], axes: &[i64]) -> Vec<usize> {
    if axes.is_empty() {
        data.iter().copied().filter(|x| *x != 1).collect()
    } else {
        let mut ans = Vec::new();
        let axes = axes
            .iter()
            .copied()
            .map(|x| if x < 0 { data.len() as i64 + x } else { x } as usize)
            .collect::<BTreeSet<_>>();
        for (i, &d) in data.iter().enumerate() {
            if axes.contains(&i) {
                if d != 1 {
                    panic!("Squeeze axes must be 1");
                }
            } else {
                ans.push(d);
            }
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
