use std::ops::{Add, Div, Mul, Sub};

pub fn pool<T>(data: &[T], kernel: &[T], dilations: &[T], pads: &[T], strides: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32>,
{
    let dim = data.len();
    debug_assert_eq!(dim, kernel.len());
    debug_assert_eq!(dim, dilations.len());
    debug_assert_eq!(dim, pads.len() / 2);
    debug_assert_eq!(dim, strides.len());

    let mut ans = Vec::with_capacity(dim);
    let (pads0, pads1) = pads.split_at(dim);
    for i in 0..dim {
        let d = data[i].clone() + pads0[i].clone() + pads1[i].clone();
        let k = (kernel[i].clone() - 1.into()) * dilations[i].clone() + 1.into();
        ans.push((d - k) / strides[i].clone() + 1.into());
    }

    ans
}

#[test]
fn test_pool() {
    assert_eq!(
        pool(&[8, 8], &[3, 3], &[2, 2], &[0, 0, 0, 0], &[1, 1],),
        &[4, 4]
    );

    assert_eq!(
        pool(&[8, 8], &[3, 3], &[1, 1], &[2, 3, 4, 5], &[1, 1],),
        &[12, 14]
    );

    assert_eq!(
        pool(&[8, 8], &[3, 3], &[1, 1], &[0, 0, 0, 0], &[1, 1],),
        &[6, 6]
    );
}
