use std::fmt::Debug;

pub fn matmul<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: PartialEq + Clone + Debug,
{
    let dim = a.len();
    debug_assert!(dim >= 2);
    debug_assert_eq!(dim, b.len());
    debug_assert_eq!(a[..dim - 2], b[..dim - 2]);
    debug_assert_eq!(a[dim - 1], b[dim - 2]);

    let mut ans = Vec::from(a);
    *ans.last_mut().unwrap() = b.last().unwrap().clone();
    ans
}

#[test]
fn test_matmul() {
    assert_eq!(matmul(&[2, 3, 4], &[2, 4, 6]), vec![2, 3, 6]);
    assert_eq!(matmul(&[1, 2, 3, 4], &[1, 2, 4, 1]), vec![1, 2, 3, 1]);
}
