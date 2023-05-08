pub fn infer(data: &[usize], shape: &[i64]) -> Vec<usize> {
    debug_assert_eq!(data.len(), shape.len());

    let mut void = None;
    let mut ans = shape
        .iter()
        .zip(data.iter())
        .enumerate()
        .map(|(i, (reshape, origin))| match *reshape {
            -1 => match void.replace(i) {
                Some(_) => panic!("Only one dimension can be inferred"),
                None => 1,
            },
            0 => *origin,
            x if x < 0 => unreachable!(),
            x => x as usize,
        })
        .collect::<Vec<_>>();
    if let Some(void) = void {
        ans[void] = data.iter().product::<usize>() / ans.iter().product::<usize>();
    } else {
        assert_eq!(
            ans.iter().product::<usize>(),
            data.iter().product::<usize>()
        );
    }
    ans
}

#[test]
fn test_rehape() {
    assert_eq!(
        infer(&[1, 2, 3, 4, 5], &[1, 0, 5, 2, -1],),
        &[1, 2, 5, 2, 6]
    );
}
