use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::Bitfield;

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

    group.throughput(Throughput::Elements(32768));
    group.bench_function("u1_load", |b| {
        b.iter(|| Bitfield::from_packed_u1::<true>(&array_u1, 0));
    });
    group.bench_function("u2_load", |b| {
        b.iter(|| Bitfield::from_packed_u2::<true>(&array_u2, 0));
    });
    group.bench_function("u4_load", |b| {
        b.iter(|| Bitfield::from_packed_u4::<true>(&array_u4, 0));
    });
    group.bench_function("u8_load", |b| {
        b.iter(|| Bitfield::from_packed_u8::<true>(&array_u8, 0));
    });
    group.bench_function("u16_load", |b| {
        b.iter(|| Bitfield::from_packed_u16::<true>(&array_u16, 0));
    });
}

criterion_group!(benches, load_data_bench);
criterion_main!(benches);
