//! QA and additive Reed-Solomon encoding with identical input shapes.

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group,
    criterion_main,
};
use p3_binary_dft::{AdditiveRsEncoder, EncodableLevel, LchNtt, PolyBasisNtt};
use p3_binary_field::{BinaryField16, BinaryField32, BinaryField64, BinaryField128 as F};
use p3_binary_qa::{
    QaEncoder, QaField, forward_transform, interleaved_encode_batch, inverse_transform,
    subfield_encode_batch,
};
use p3_commit::Encoder;
use p3_field::{ExtensionField, PrimeCharacteristicRing};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const LOG_BATCH: usize = 5;

fn bench_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    group
}

fn random<T>(len: usize, seed: u64) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random()).collect()
}

fn qa(log_k: usize, rate: usize) -> QaEncoder<F> {
    let count = ((1 << rate) - 1) << log_k;
    QaEncoder::new(log_k, rate, random(count, 92))
}

fn qa_bf64(log_k: usize) -> QaEncoder<F, BinaryField64> {
    QaEncoder::new(log_k, 2, random(3 << log_k, 92))
}

fn bench_interleaved_bf128(c: &mut Criterion) {
    let mut group = bench_group(c, "bench_interleaved_bf128");
    let lch = AdditiveRsEncoder::<F, LchNtt<F>>::default();
    let poly = AdditiveRsEncoder::<F, PolyBasisNtt>::default();
    let width = 1 << LOG_BATCH;
    for log_n in [20, 22, 24] {
        let log_k = log_n - LOG_BATCH;
        let input = RowMajorMatrix::new(random::<F>(1 << log_n, 128), width);
        group.throughput(Throughput::Bytes(
            size_of_val(input.values.as_slice()) as u64
        ));
        for rate in [1, 2] {
            let case = format!("n{log_n}_h{LOG_BATCH}_c{}", 1 << rate);
            let encoder = qa(log_k, rate);
            group.bench_function(BenchmarkId::new("qa_full", &case), |b| {
                b.iter(|| interleaved_encode_batch(&encoder, black_box(&input.values), LOG_BATCH));
            });
            group.bench_function(BenchmarkId::new("rs_lch", &case), |b| {
                b.iter(|| lch.encode_batch_borrowed(black_box(input.as_view()), rate));
            });
            group.bench_function(BenchmarkId::new("rs_poly", &case), |b| {
                b.iter(|| poly.encode_batch_borrowed(black_box(input.as_view()), rate));
            });
            if rate == 2 {
                let encoder = qa_bf64(log_k);
                group.bench_function(BenchmarkId::new("qa_bf64_mult", &case), |b| {
                    b.iter(|| {
                        interleaved_encode_batch(&encoder, black_box(&input.values), LOG_BATCH)
                    });
                });
            }
        }
    }
}

fn widen<S: Copy + Send + Sync>(input: &[S]) -> RowMajorMatrix<F>
where
    F: From<S>,
{
    RowMajorMatrix::new(
        input.par_iter().copied().map(F::from).collect(),
        1 << LOG_BATCH,
    )
}

fn bench_witness<S>(group: &mut BenchmarkGroup<'_, WallTime>, bits: usize)
where
    S: QaField + EncodableLevel,
    F: ExtensionField<S>,
    StandardUniform: Distribution<S>,
{
    let lch = AdditiveRsEncoder::<F, LchNtt<F>>::default();
    let poly = AdditiveRsEncoder::<F, PolyBasisNtt>::default();
    let native = S::Encoder::default();
    for log_n in [22, 24] {
        let input = random::<S>(1 << log_n, bits as u64);
        let log_k = log_n - LOG_BATCH;
        group.throughput(Throughput::Bytes(size_of_val(input.as_slice()) as u64));
        for rate in [1, 2] {
            let case = format!("bf{bits}_n{log_n}_h{LOG_BATCH}_c{}", 1 << rate);
            let encoder = qa(log_k, rate);
            group.bench_function(BenchmarkId::new("qa_mixed", &case), |b| {
                b.iter(|| subfield_encode_batch(&encoder, black_box(&input), LOG_BATCH));
            });
            group.bench_function(BenchmarkId::new("widen_then_rs_lch", &case), |b| {
                b.iter(|| {
                    let wide = widen(black_box(&input));
                    lch.encode_batch_borrowed(wide.as_view(), rate)
                });
            });
            group.bench_function(BenchmarkId::new("widen_then_rs_poly", &case), |b| {
                b.iter(|| {
                    let wide = widen(black_box(&input));
                    poly.encode_batch_borrowed(wide.as_view(), rate)
                });
            });
            // A native narrow RS transform is equivalent while the Cantor
            // domain lies in the subfield. Widen its output to match the API.
            if log_k + rate <= 1 << S::LOG_BITS {
                group.bench_function(BenchmarkId::new("native_rs_then_widen", &case), |b| {
                    b.iter(|| {
                        let view = RowMajorMatrixView::new(black_box(&input), 1 << LOG_BATCH);
                        let narrow = native.encode_batch_borrowed(view, rate);
                        widen(&narrow.values)
                    });
                });
            }
        }
    }
}

fn bench_subfield_to_bf128(c: &mut Criterion) {
    let mut group = bench_group(c, "bench_subfield_to_bf128");
    bench_witness::<BinaryField16>(&mut group, 16);
    bench_witness::<BinaryField32>(&mut group, 32);
    bench_witness::<BinaryField64>(&mut group, 64);
}

fn rs_rows(
    encoder: &impl Encoder<F>,
    message: &RowMajorMatrix<F>,
    rate: usize,
) -> RowMajorMatrix<F> {
    let interleaved = message.transpose_packable();
    encoder
        .encode_batch_borrowed(interleaved.as_view(), rate)
        .transpose_packable()
}

fn bench_square_matrix(c: &mut Criterion) {
    let mut group = bench_group(c, "bench_square_matrix");
    let lch = AdditiveRsEncoder::<F, LchNtt<F>>::default();
    let poly = AdditiveRsEncoder::<F, PolyBasisNtt>::default();
    for log_k in [11, 12] {
        let input = RowMajorMatrix::new(random::<F>(1 << (2 * log_k), 128), 1 << log_k);
        group.throughput(Throughput::Bytes(
            size_of_val(input.values.as_slice()) as u64
        ));
        for rate in [1, 2] {
            let case = format!("n{}_h{log_k}_c{}", 2 * log_k, 1 << rate);
            let encoder = qa(log_k, rate);
            group.bench_function(BenchmarkId::new("qa_full", &case), |b| {
                b.iter(|| encoder.encode_batch(black_box(&input)));
            });
            group.bench_function(BenchmarkId::new("rs_lch", &case), |b| {
                b.iter(|| rs_rows(&lch, black_box(&input), rate));
            });
            group.bench_function(BenchmarkId::new("rs_poly", &case), |b| {
                b.iter(|| rs_rows(&poly, black_box(&input), rate));
            });
            if rate == 2 {
                let encoder = qa_bf64(log_k);
                group.bench_function(BenchmarkId::new("qa_bf64_mult", &case), |b| {
                    b.iter(|| encoder.encode_batch(black_box(&input)));
                });
            }
        }
    }
}

fn unfused(encoder: &QaEncoder<F>, input: &[F], width: usize) -> Vec<F> {
    let mut output = F::zero_vec(2 * input.len());
    let (systematic, parity) = output.split_at_mut(input.len());
    let mut spectrum = F::zero_vec(input.len());
    systematic
        .par_chunks_mut(4096)
        .zip(spectrum.par_chunks_mut(4096))
        .zip(input.par_chunks(4096))
        .for_each(|((systematic, spectrum), input)| {
            systematic.copy_from_slice(input);
            spectrum.copy_from_slice(input);
        });
    forward_transform(&mut spectrum, width);
    encoder.multiply_spectrum(&spectrum, width, 0, parity);
    inverse_transform(parity, width);
    output
}

fn bench_stage_breakdown(c: &mut Criterion) {
    const LOG_N: usize = 22;
    let width = 1 << LOG_BATCH;
    let encoder = qa(LOG_N - LOG_BATCH, 1);
    let input = random::<F>(1 << LOG_N, 128);
    let mut spectrum = input.clone();
    forward_transform(&mut spectrum, width);
    let mut product = F::zero_vec(input.len());
    encoder.multiply_spectrum(&spectrum, width, 0, &mut product);

    let mut group = bench_group(c, "bench_stage_breakdown");
    group.throughput(Throughput::Bytes(size_of_val(input.as_slice()) as u64));
    group.bench_function("forward", |b| {
        b.iter_batched_ref(
            || input.clone(),
            |values| forward_transform(black_box(values), width),
            BatchSize::PerIteration,
        );
    });
    group.bench_function("pointwise", |b| {
        let mut output = F::zero_vec(input.len());
        b.iter(|| {
            encoder.multiply_spectrum(black_box(&spectrum), width, 0, black_box(&mut output));
        });
    });
    group.bench_function("inverse", |b| {
        b.iter_batched_ref(
            || product.clone(),
            |values| inverse_transform(black_box(values), width),
            BatchSize::PerIteration,
        );
    });
    group.bench_function("unfused", |b| {
        b.iter(|| unfused(&encoder, black_box(&input), width));
    });
    group.bench_function("v_cycle", |b| {
        b.iter(|| encoder.interleaved_encode_batch(black_box(&input), LOG_BATCH));
    });
}

criterion_group!(
    benches,
    bench_interleaved_bf128,
    bench_subfield_to_bf128,
    bench_square_matrix,
    bench_stage_breakdown
);
criterion_main!(benches);
