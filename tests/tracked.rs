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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

    untracked1.andnot_assign(&untracked2);
    tracked1.andnot_assign(&tracked2);

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_shr_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);

    untracked1 >>= 12;
    tracked1 >>= 12;

    assert_eq!(untracked1, *tracked1.as_bitfield());
}

#[test]
fn tracked_shl_test() {
    let array1 = gen_sparse::<2048>(0, NUM_BATCHES, 6, 60);

    let mut untracked1 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);

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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

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

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);

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

    let mut untracked1 = Bitfield::new(0);
    let mut untracked2 = Bitfield::new(0);
    let mut untracked3 = Bitfield::new(0);
    let mut untracked4 = Bitfield::new(0);
    let mut untracked5 = Bitfield::new(0);
    let mut untracked6 = Bitfield::new(0);
    let mut tracked1 = TrackedBitfield::new(0);
    let mut tracked2 = TrackedBitfield::new(0);
    let mut tracked3 = TrackedBitfield::new(0);
    let mut tracked4 = TrackedBitfield::new(0);
    let mut tracked5 = TrackedBitfield::new(0);
    let mut tracked6 = TrackedBitfield::new(0);

    untracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    untracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    untracked3.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array3, 0);
    untracked4.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array4, 0);
    untracked5.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array5, 0);
    untracked6.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array6, 0);
    tracked1.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array1, 0);
    tracked2.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array2, 0);
    tracked3.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array3, 0);
    tracked4.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array4, 0);
    tracked5.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array5, 0);
    tracked6.load_packed_u4_into::<SET_FLAG_OR, CMP_FLAG_NE>(&array6, 0);

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

    assert_eq!(untracked1, *tracked1.as_bitfield());
}
