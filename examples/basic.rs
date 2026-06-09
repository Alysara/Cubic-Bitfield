use std::hint::black_box;

use cubic_bitfields::*;

fn main() {
    let bitfield = Bitfield::new(1);
    // black_box(&bitfield);
    //
    // for z in 0..32 {
    //     for y in 0..32 {
    //         bitfield.as_mut_slice()[z * 32 + y] = u32::MAX >> (31 - y);
    //     }
    // }
    //
    // for i in 0..1024 {
    //     bitfield.as_mut_slice()[i] = (i * 12341) as u32;
    // }

    // for z in 0..32 {
    //     for y in (z + 1)..32 {
    //         bitfield.as_mut_slice()[y * 32 + z] = u32::MAX;
    //     }
    // }
    let mut bitfield = Bitfield::new(0);

    let array_u1: [u64; 512] = std::array::from_fn(|_| 0xF0F0F0F0F0F0F0F0);
    let array_u2: [u64; 1024] = std::array::from_fn(|_| 0xF0F0F0F0F0F0F0F0);
    let array_u4: [u64; 2048] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    let array_u8: [u64; 4096] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    let array_u16: [u64; 8192] = std::array::from_fn(|_| 0x0000FFFF0000FFFF);
    // // let bitfield = black_box(Bitfield::from_packed_u16::<true>(&array_u16, black_box(0)));
    // // let bitfield = black_box(Bitfield::from_packed_u8::<true>(&array_u8, black_box(0)));
    // let bitfield1 = black_box(Bitfield::from_packed_u1::<true>(&array_u1, black_box(0)));
    bitfield.load_packed_u2_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u2, 0);
    // let bitfield3 = black_box(Bitfield::from_packed_u4::<true>(&array_u4, black_box(0)));
    // let bitfield4 = black_box(Bitfield::from_packed_u8::<true>(&array_u8, black_box(0)));
    // let bitfield5 = black_box(Bitfield::from_packed_u16::<true>(&array_u16, black_box(0)));
    //
    // let bitfield1 = Bitfield::new(2);

    // black_box(black_box(bitfield1) ^ black_box(bitfield));

    black_box(bitfield);
    // bitfield.print_inner_slices(0..32);
}
