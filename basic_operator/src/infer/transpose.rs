use std::collections::BTreeMap;

// 1st: new shape
// 2nd: permutation
pub fn infer<T, U>(shape: &[T], src: &[U], tgt: &[U]) -> (Vec<T>, Vec<i32>)
where
    T: Clone,
    U: Ord,
{
    assert_eq!(shape.len(), src.len());
    assert_eq!(shape.len(), tgt.len());

    let mut indices = BTreeMap::new();
    for (i, c) in src.iter().enumerate() {
        indices.insert(c, i);
    }
    let mut ans = (
        Vec::with_capacity(shape.len()),
        Vec::with_capacity(shape.len()),
    );
    for c in tgt {
        let idx = indices[&c];
        ans.0.push(shape[idx].clone());
        ans.1.push(idx as i32);
    }
    ans
}

#[test]
fn test_transpose() {
    assert_eq!(
        infer(&[1, 2, 3, 4], "nchw".as_bytes(), "nhwc".as_bytes()),
        (vec![1, 3, 4, 2], vec![0, 2, 3, 1])
    );
}
