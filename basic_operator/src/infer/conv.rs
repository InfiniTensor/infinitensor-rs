use std::ops::{Add, Div, Mul, Sub};

pub fn conv<T>(data: &[T], kernel: &[T], dilations: &[T], pads: &[T], strides: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32>,
{
    let dim = data.len();
    debug_assert_eq!(dim, kernel.len());
    debug_assert_eq!(dim, dilations.len() + 2);
    debug_assert_eq!(dim, pads.len() / 2 + 2);
    debug_assert_eq!(dim, strides.len() + 2);

    let mut ans = Vec::with_capacity(data.len());
    ans.push(data[0].clone());
    ans.push(kernel[0].clone());

    let dim = dim - 2;
    let data = &data[2..];
    let kernel = &kernel[2..];
    let (pads0, pads1) = pads.split_at(dim);

    for i in 0..dim {
        let d = data[i].clone() + pads0[i].clone() + pads1[i].clone();
        let k = kernel[i].clone() * dilations[i].clone() - 1.into();
        ans.push((d - k + 1.into()) / strides[i].clone());
    }

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
        &[7, 6, 13, 15]
    );

    assert_eq!(
        conv(
            &[7, 8, 8, 8],
            &[6, 2, 3, 3],
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
        ),
        &[7, 6, 7, 7]
    );
}
