use std::hint::black_box;

use cubic_bitfield::*;

fn main() {
    let mut bitfield = Bitfield::new(1);
    black_box(&bitfield);

    for z in 0..32 {
        for y in 0..32 {
            bitfield.as_mut_slice()[z * 32 + y] = u32::MAX >> (31 - y);
        }
    }

    for i in 0..1024 {
        bitfield.as_mut_slice()[i] = (i * 12341) as u32;
    }

    // for z in 0..32 {
    //     for y in (z + 1)..32 {
    //         bitfield.as_mut_slice()[y * 32 + z] = u32::MAX;
    //     }
    // }

    println!("Input:");
    bitfield.print_inner_slices(0..1);

    bitfield.inner_transpose_scalar();
    black_box(&bitfield);

    println!("Output:");
    bitfield.print_inner_slices(0..1);

    // bitfield.print_outer_slices(0..3);

    bitfield.outer_transpose();
    black_box(&bitfield);
}
