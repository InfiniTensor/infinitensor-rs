use super::pool;
use std::ops::{Add, Div, Mul, Sub};

pub fn conv<T>(data: &[T], kernel: &[T], dilations: &[T], pads: &[T], strides: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32>,
{
    let dim = data.len();
    debug_assert!(dim >= 2);
    debug_assert_eq!(dim, kernel.len());

    let mut ans = Vec::with_capacity(dim);
    ans.push(data[0].clone());
    ans.push(kernel[0].clone());
    ans.extend(pool(&data[2..], &kernel[2..], dilations, pads, strides));
    ans
}

#[test]
fn test_conv() {
    assert_eq!(
        conv(
            &[7, 3, 8, 8],
            &[6, 3, 3, 3],
            &[2, 2],
            &[0, 0, 0, 0],
            &[1, 1],
        ),
        &[7, 6, 4, 4]
    );

    assert_eq!(
        conv(
            &[7, 3, 8, 8],
            &[6, 3, 3, 3],
            &[1, 1],
            &[2, 3, 4, 5],
            &[1, 1],
        ),
        &[7, 6, 12, 14]
    );

    assert_eq!(
        conv(
            &[7, 8, 8, 8],
            &[6, 2, 3, 3],
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
        ),
        &[7, 6, 6, 6]
    );
}
