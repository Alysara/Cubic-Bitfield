use std::ops::Range;

pub fn print_matrix_inner_slices(matrix: &[u32; 1024], range: Range<usize>) {
    assert!(
        range.end <= 32,
        "End of range is too large! {} is not <= 32.",
        range.end
    );
    assert!(
        range.start < 32,
        "Start of range is too large! {} is not < 32.",
        range.start
    );

    for slice in range {
        println!("{:-<35}", format!("|- Inner slice {slice} "));

        for i in 0..32 {
            let index = slice * 32 + i;
            let prefix = if i % 2 == 0 { "+" } else { "|" };
            println!("{prefix} {:032b}", matrix[index]);
        }
        println!("{:-<35}", "|");
        println!();
    }
}

pub fn print_matrix_outer_slices(matrix: &[u32; 1024], range: Range<usize>) {
    assert!(
        range.end <= 32,
        "End of range is too large! {} is not <= 32.",
        range.end
    );
    assert!(
        range.start < 32,
        "Start of range is too large! {} is not < 32.",
        range.start
    );

    for slice in range {
        println!("{:-<35}", format!("|- Outer slice {slice} "));

        for i in 0..32 {
            let mut bits = 0;
            for j in 0..32 {
                let index = i * 32 + j;
                let bit = (matrix[index] >> slice) & 1;
                bits |= bit << j
            }
            let prefix = if i % 2 == 0 { "+" } else { "|" };
            println!("{prefix} {:032b}", bits);
        }
        println!("{:-<35}", "|");
        println!();
    }
}
