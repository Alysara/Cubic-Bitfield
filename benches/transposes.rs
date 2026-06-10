use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn transpose_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitfield");
    let mut bitfield = Bitfield::new(12);
    black_box(bitfield);

    group.throughput(Throughput::Elements(1024));
    group.bench_function("inner_transpose", |b| {
        b.iter(|| {
            bitfield.inner_transpose();
            black_box(bitfield);
        });
    });
    group.bench_function("outer_transpose", |b| {
        b.iter(|| {
            bitfield.outer_transpose();
            black_box(bitfield);
        });
    });
    group.bench_function("outer_transpose_scalar", |b| {
        b.iter(|| {
            bitfield.outer_transpose_scalar();
            black_box(bitfield);
        });
    });
    group.finish();
}

fn tracked_bitfield_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracked_bitfield");
    let mut bitfield1 = Bitfield::new(0);
    let mut bitfield2 = Bitfield::new(0);

    const NUM_BATCHES: usize = 1;

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

    group.throughput(Throughput::Elements(1024));
    // group.bench_function("untracked_and", |b| {
    //     b.iter(|| untracked1 & untracked2);
    // });
    // group.bench_function("tracked_and", |b| {
    //     b.iter(|| tracked1 & tracked2);
    // });
    //
    // group.bench_function("untracked_or", |b| {
    //     b.iter(|| {
    //         untracked1 |= untracked2;
    //         black_box(&untracked1);
    //     });
    // });
    // group.bench_function("tracked_or", |b| {
    //     b.iter(|| {
    //         tracked1 |= tracked2;
    //         black_box(&tracked1);
    //     });
    // });

    group.bench_function("untracked_xor", |b| {
        b.iter(|| {
            untracked1 ^= untracked2;
            black_box(&untracked1);
        });
    });
    group.bench_function("tracked_xor", |b| {
        b.iter(|| {
            tracked1 ^= tracked2;
            black_box(&tracked1);
        });
    });

    group.bench_function("untracked_inner_transpose", |b| {
        b.iter(|| {
            black_box(untracked1.inner_transpose());
        });
    });
    group.bench_function("tracked_inner_transpose", |b| {
        b.iter(|| {
            black_box(tracked1.inner_transpose());
        });
    });
    group.finish();
}

fn load_data_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_bitfield");
    let array_u1: [u64; 512] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u2: [u64; 1024] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u4: [u64; 2048] = std::array::from_fn(|_| 0x0F0F0F0F0F0F0F0F);
    let array_u8: [u64; 4096] = std::array::from_fn(|_| 0x00FF00FF00FF00FF);
    let array_u16: [u64; 8192] = std::array::from_fn(|_| 0x0000FFFF0000FFFF);

    black_box(array_u1);
    black_box(array_u2);
    black_box(array_u4);
    black_box(array_u8);
    black_box(array_u16);

    let mut bitfield = Bitfield::new(0);
    group.throughput(Throughput::Elements(32768));
    group.bench_function("u1_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u1_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u1, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u2_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u2, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u4_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u4, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u8_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u8, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_load", |b| {
        b.iter(|| {
            bitfield.load_packed_u16_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u16, 0);
            black_box(&bitfield);
        });
    });

    group.bench_function("u1_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u1_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u1, 31, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u2_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u2, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u4_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u4, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u8_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u8, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_yz_load", |b| {
        b.iter(|| {
            bitfield.load_yz_packed_u16_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u16, 31, 0);
            black_box(&bitfield);
        });
    });

    group.bench_function("u1_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u1_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u1, 31, false);
            black_box(&bitfield);
        });
    });
    group.bench_function("u2_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u2_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u2, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u4_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u4_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u4, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u8_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u8_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u8, 31, 0);
            black_box(&bitfield);
        });
    });
    group.bench_function("u16_xz_load", |b| {
        b.iter(|| {
            bitfield.load_xz_packed_u16_into::<SET_FLAG_ASSIGN, CMP_FLAG_EQ>(&array_u16, 31, 0);
            black_box(&bitfield);
        });
    });
}

criterion_group!(benches, tracked_bitfield_bench);
criterion_main!(benches);
