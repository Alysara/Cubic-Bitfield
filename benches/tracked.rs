use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubic_bitfields::util::gen_sparse;
use cubic_bitfields::*;

fn tracked_bitfield_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracked_bitfield");

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

    group.bench_function("update_tracking", |b| {
        b.iter(|| {
            black_box(&tracked1);
            tracked1.update_tracking();
            black_box(&tracked1);
        });
    });
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

criterion_group!(benches, tracked_bitfield_bench);
criterion_main!(benches);
