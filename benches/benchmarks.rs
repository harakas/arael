use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arael::utils::Float;

fn bench_system_atan(c: &mut Criterion) {
    c.bench_function("system_atan", |b| b.iter(|| {
        let max_value = 10000.0;
        let iters = 100000;
        for n in -iters..=iters {
            let x = n as f64 / iters as f64 * max_value;
            black_box(x.atan());
        }
    }));
}

fn bench_fast_atan(c: &mut Criterion) {
    c.bench_function("fast_atan", |b| b.iter(|| {
        let max_value = 10000.0;
        let iters = 100000;
        for n in -iters..=iters {
            let x = n as f64 / iters as f64 * max_value;
            black_box(x.fast_atan());
        }
    }));
}

fn bench_system_atan2(c: &mut Criterion) {
    c.bench_function("system_atan2", |b| b.iter(|| {
        let max_value = 10.0;
        let iters = 1000;
        for m in -iters..=iters {
            let y = m as f64 / iters as f64 * max_value;
            for n in -iters..=iters {
                let x = n as f64 / iters as f64 * max_value;
                black_box(arael::utils::atan2(y, x));
            }
        }
    }));
}

fn bench_fast_atan2(c: &mut Criterion) {
    c.bench_function("fast_atan2", |b| b.iter(|| {
        let max_value = 10.0;
        let iters = 1000;
        for m in -iters..=iters {
            let y = m as f64 / iters as f64 * max_value;
            for n in -iters..=iters {
                let x = n as f64 / iters as f64 * max_value;
                black_box(arael::utils::fast_atan2(y, x));
            }
        }
    }));
}

criterion_group!(benches, bench_system_atan, bench_fast_atan, bench_system_atan2, bench_fast_atan2);
criterion_main!(benches);
