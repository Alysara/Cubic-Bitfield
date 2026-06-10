use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::{Bitfield, *};

const NUM_BATCHES: usize = 6;

#[test]
fn tracked_and_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1 &= untracked2;
    tracked1 &= tracked2;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_or_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1 |= untracked2;
    tracked1 |= tracked2;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_xor_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1 ^= untracked2;
    tracked1 ^= tracked2;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_andnot_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1.andnot_assign(&untracked2);
    tracked1.andnot_assign(&tracked2);

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_shr_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);

    untracked1 >>= 12;
    tracked1 >>= 12;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_shl_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);

    untracked1 <<= 12;
    tracked1 <<= 12;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_inner_transpose_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, 1000, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1.inner_transpose();
    untracked2.inner_transpose();
    tracked1.inner_transpose();
    tracked2.inner_transpose();

    assert_eq!(untracked1, *tracked1.as_bitfield());
    assert_eq!(untracked2, *tracked2.as_bitfield());
}

#[test]
fn tracked_outer_transpose_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, 1000, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);

    untracked1.outer_transpose();
    untracked2.outer_transpose();
    tracked1.outer_transpose();
    tracked2.outer_transpose();

    assert_eq!(untracked1, *tracked1.as_bitfield());
    assert_eq!(untracked2, *tracked2.as_bitfield());
}

#[test]
fn tracked_multi_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let array2 = gen_sparse::<2048>(1, NUM_BATCHES, 6, 60);
    let array3 = gen_sparse::<2048>(2, NUM_BATCHES, 6, 60);
    let array4 = gen_sparse::<2048>(3, NUM_BATCHES, 6, 60);
    let array5 = gen_sparse::<2048>(4, NUM_BATCHES, 6, 60);
    let array6 = gen_sparse::<2048>(5, NUM_BATCHES, 6, 60);
    let array7 = gen_sparse::<2048>(6, 1000, 6, 60);
    let array8 = gen_sparse::<2048>(7, 1000, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut untracked3 = Bitfield::new(0);
    let mut untracked4 = Bitfield::new(0);
    let mut untracked5 = Bitfield::new(0);
    let mut untracked6 = Bitfield::new(0);
    let mut untracked7 = Bitfield::new(0);
    let mut untracked8 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);
    let mut tracked3 = TrackedBitfield::new(0);
    let mut tracked4 = TrackedBitfield::new(0);
    let mut tracked5 = TrackedBitfield::new(0);
    let mut tracked6 = TrackedBitfield::new(0);
    let mut tracked7 = TrackedBitfield::new(0);
    let mut tracked8 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    untracked3.load_packed_u4_into::<SET_OR, CMP_NE>(&array3, 0);
    untracked4.load_packed_u4_into::<SET_OR, CMP_NE>(&array4, 0);
    untracked5.load_packed_u4_into::<SET_OR, CMP_NE>(&array5, 0);
    untracked6.load_packed_u4_into::<SET_OR, CMP_NE>(&array6, 0);
    untracked7.load_packed_u4_into::<SET_OR, CMP_NE>(&array7, 0);
    untracked8.load_packed_u4_into::<SET_OR, CMP_NE>(&array8, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_OR, CMP_NE>(&array2, 0);
    tracked3.load_packed_u4_into::<SET_OR, CMP_NE>(&array3, 0);
    tracked4.load_packed_u4_into::<SET_OR, CMP_NE>(&array4, 0);
    tracked5.load_packed_u4_into::<SET_OR, CMP_NE>(&array5, 0);
    tracked6.load_packed_u4_into::<SET_OR, CMP_NE>(&array6, 0);
    tracked7.load_packed_u4_into::<SET_OR, CMP_NE>(&array7, 0);
    tracked8.load_packed_u4_into::<SET_OR, CMP_NE>(&array8, 0);

    untracked1 &= untracked2;
    tracked1 &= tracked2;

    untracked1 ^= untracked3;
    tracked1 ^= tracked3;

    untracked1.inner_transpose();
    tracked1.inner_transpose();

    untracked1 |= untracked4;
    tracked1 |= tracked4;

    untracked1.outer_transpose().inner_transpose();
    tracked1.outer_transpose().inner_transpose();

    untracked1.andnot_assign(&untracked5);
    tracked1.andnot_assign(&tracked5);

    untracked1 &= untracked6;
    tracked1 &= tracked6;

    untracked1 ^= untracked7;
    tracked1 ^= tracked7;

    untracked1 &= untracked8;
    tracked1 &= tracked8;

    untracked1.inner_transpose().outer_transpose();
    tracked1.inner_transpose().outer_transpose();

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn active_bit_iter_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);
    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    untracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_OR, CMP_NE>(&array1, 0);
    assert_eq!(
        untracked1,
        *tracked1.as_bitfield(),
        "Bitfield as not equal."
    );

    let mut iter = tracked1.active_bit_iter();

    let true_array = untracked1.as_array();

    for (i, cur_entry) in true_array.iter().enumerate() {
        let mut entry = *cur_entry;
        while entry != 0 {
            let bit = entry.trailing_zeros();
            let val = i * 32 + bit as usize;

            let iter_val = iter.next().expect("No next iter value??");
            assert_eq!(val, iter_val);

            entry &= entry - 1;
        }
    }
}
