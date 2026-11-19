//! Natural-order tensor transforms. Every layer acts on a different index bit.

use core::{array, ptr};

use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{QaButterflyValue, QaField};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type Simd = p3_binary_field::M512;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
type Simd = p3_binary_field::M256;
#[cfg(not(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
)))]
type Simd = p3_binary_field::M128;

/// Bound the wider buffer of a cache leaf to about 64 KiB.
pub(crate) const CACHE_BYTES: usize = 64 * 1024;
const PASS_GRAIN: usize = 128 * 1024;
const RADIX16_MIN_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn cache_log<F>(width: usize, log_len: usize) -> usize {
    cache_log_for_bytes::<F>(width, log_len, CACHE_BYTES)
}

fn cache_log_for_bytes<F>(width: usize, log_len: usize, bytes: usize) -> usize {
    let rows = (bytes / size_of::<F>() / width).max(1);
    (rows.ilog2() as usize).min(log_len)
}

/// Use a larger leaf only if it removes a high-stride pass.
pub(crate) fn encoding_cache_log<F, const PARALLEL: bool>(width: usize, log_len: usize) -> usize {
    let small = cache_log::<F>(width, log_len);
    let large = cache_log_for_bytes::<F>(width, log_len, 4 * CACHE_BYTES);
    // A remainder uses the original radix-four and radix-two kernels.
    let passes = |bottom| {
        let layers: usize = log_len - bottom;
        layers / 4 + (layers % 4).div_ceil(2)
    };
    if PARALLEL
        && passes(large) < passes(small)
        && ((width << log_len) * size_of::<F>()) >= RADIX16_MIN_BYTES
        && current_num_threads() > 1
    {
        large
    } else {
        small
    }
}

/// Apply all forward CRT layers to an interleaved buffer in natural order.
///
/// The width is the number of messages and each coordinate occupies one row.
/// The number of coordinates must be a power of two.
pub fn forward_transform<F: QaField>(values: &mut [F], width: usize) {
    transform::<F, false>(values, width);
}

/// Invert the forward CRT transform in place.
pub fn inverse_transform<F: QaField>(values: &mut [F], width: usize) {
    transform::<F, true>(values, width);
}

fn transform<F: QaField, const INVERSE: bool>(values: &mut [F], width: usize) {
    assert!(width != 0 && values.len().is_multiple_of(width));
    let log_len = log2_strict_usize(values.len() / width);
    let bottom = cache_log::<F>(width, log_len);
    let leaves = |values: &mut [F]| {
        values
            .par_chunks_exact_mut(width << bottom)
            .with_min_len(2)
            .for_each(|leaf| layers::<F, INVERSE, false>(leaf, width, 0, bottom));
    };
    if INVERSE {
        leaves(values);
        layers::<F, true, true>(values, width, bottom, log_len);
    } else {
        layers::<F, false, true>(values, width, bottom, log_len);
        leaves(values);
    }
}

/// Fuse four high-stride layers in a single pass. Sixteen runs feed each
/// register tile, so the intermediate layers do not write out the matrix.
pub(crate) fn upper_layers<F: QaField, const INVERSE: bool, const PARALLEL: bool>(
    values: &mut [F],
    width: usize,
    mut lo: usize,
    mut hi: usize,
) {
    if !PARALLEL || size_of_val(values) < RADIX16_MIN_BYTES || current_num_threads() == 1 {
        layers::<F, INVERSE, PARALLEL>(values, width, lo, hi);
        return;
    }
    while hi - lo >= 4 {
        let bit = if INVERSE {
            let bit = lo;
            lo += 4;
            bit
        } else {
            hi -= 4;
            hi
        };
        pass::<F, INVERSE, PARALLEL, 16>(values, width << bit);
    }
    layers::<F, INVERSE, PARALLEL>(values, width, lo, hi);
}

/// Apply index bits in the half-open range lo..hi, pairing adjacent layers.
pub(crate) fn layers<F: QaField, const INVERSE: bool, const PARALLEL: bool>(
    values: &mut [F],
    width: usize,
    mut lo: usize,
    mut hi: usize,
) {
    while hi - lo >= 2 {
        let bit = if INVERSE {
            let bit = lo;
            lo += 2;
            bit
        } else {
            hi -= 2;
            hi
        };
        pass::<F, INVERSE, PARALLEL, 4>(values, width << bit);
    }
    if hi != lo {
        pass::<F, INVERSE, PARALLEL, 2>(values, width << lo);
    }
}

fn pass<F: QaField, const INVERSE: bool, const PARALLEL: bool, const N: usize>(
    values: &mut [F],
    span: usize,
) {
    let group_size = N * span;
    if PARALLEL && size_of_val(values) >= 2 * PASS_GRAIN {
        let groups = values.len() / group_size;
        let stripes = current_num_threads().div_ceil(groups);
        values
            .par_chunks_exact_mut(group_size)
            .with_min_len((PASS_GRAIN / size_of::<F>() / group_size).max(1))
            .for_each(|group| {
                let mut runs = group.chunks_exact_mut(span);
                striped::<F, INVERSE, N>(array::from_fn(|_| runs.next().unwrap()), stripes);
            });
    } else {
        for group in values.chunks_exact_mut(group_size) {
            let mut runs = group.chunks_exact_mut(span);
            packed_butterflies::<F, INVERSE, N>(array::from_fn(|_| runs.next().unwrap()));
        }
    }
}

/// A high-stride group can itself occupy the whole matrix. Split its independent lanes.
fn striped<F: QaField, const INVERSE: bool, const N: usize>(runs: [&mut [F]; N], stripes: usize) {
    if stripes > 1 && N * size_of_val(runs[0]) >= 2 * PASS_GRAIN {
        let split = runs[0].len() / 2;
        let mut right = array::from_fn(|_| &mut [][..]);
        let mut runs = runs.into_iter();
        let left = array::from_fn(|i| {
            let (left, rest) = runs.next().unwrap().split_at_mut(split);
            right[i] = rest;
            left
        });
        let half = stripes.div_ceil(2);
        join(
            || striped::<F, INVERSE, N>(left, half),
            || striped::<F, INVERSE, N>(right, half),
        );
    } else {
        packed_butterflies::<F, INVERSE, N>(runs);
    }
}

#[inline(always)]
fn butterfly<V: QaButterflyValue, const INVERSE: bool, const N: usize>(values: &mut [V; N]) {
    match values.as_mut_slice() {
        [x, y] if INVERSE => V::butterfly_u(x, y),
        [x, y] => V::butterfly_t(x, y),
        [a, b, c, d] if INVERSE => V::butterfly_u_radix4(a, b, c, d),
        [a, b, c, d] => V::butterfly_t_radix4(a, b, c, d),
        lanes if N == 16 => butterfly_radix16::<_, INVERSE>(lanes),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn butterfly_radix16<V: QaButterflyValue, const INVERSE: bool>(lanes: &mut [V]) {
    let ([row0, row1, row2, row3], []) = lanes.as_chunks_mut::<4>() else {
        unreachable!();
    };
    if INVERSE {
        for row in [&mut *row0, &mut *row1, &mut *row2, &mut *row3] {
            butterfly::<_, true, 4>(row);
        }
        for i in 0..4 {
            V::butterfly_u_radix4(&mut row0[i], &mut row1[i], &mut row2[i], &mut row3[i]);
        }
    } else {
        for i in 0..4 {
            V::butterfly_t_radix4(&mut row0[i], &mut row1[i], &mut row2[i], &mut row3[i]);
        }
        for row in [row0, row1, row2, row3] {
            butterfly::<_, false, 4>(row);
        }
    }
}

#[inline(always)]
fn packed_butterflies<F: QaField, const INVERSE: bool, const N: usize>(runs: [&mut [F]; N]) {
    let lanes = size_of::<Simd>() / size_of::<F>();
    let len = runs[0].len();
    let packed_len = len / lanes * lanes;
    let mut chunks = runs.map(|run| run.chunks_exact_mut(lanes));
    for _ in 0..packed_len / lanes {
        let blocks = chunks.each_mut().map(|iter| iter.next().unwrap());
        let mut packed = blocks.each_ref().map(|block| {
            // SAFETY: a block is exactly one Simd word array in bytes. QaField
            // is sealed to transparent, full-width integers. The load is unaligned.
            unsafe { ptr::read_unaligned(block.as_ptr().cast::<Simd>()) }
        });
        butterfly::<_, INVERSE, N>(&mut packed);
        for (block, word) in blocks.into_iter().zip(packed) {
            // SAFETY: the same block remains uniquely borrowed, and every
            // resulting integer bit pattern is a valid tower element.
            unsafe { ptr::write_unaligned(block.as_mut_ptr().cast::<Simd>(), word) };
        }
    }
    let mut tails = chunks.map(|chunks| chunks.into_remainder());
    for index in 0..len - packed_len {
        let mut values = tails.each_ref().map(|run| run[index]);
        butterfly::<_, INVERSE, N>(&mut values);
        for (run, value) in tails.iter_mut().zip(values) {
            run[index] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::BinaryField8 as F;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::{naive_forward, naive_inverse};

    #[test]
    fn four_layer_pass_matches_paired_layers() {
        let mut rng = SmallRng::seed_from_u64(98);
        for width in [1, 4, 32, 128] {
            let original: Vec<F> = (0..16 * width).map(|_| rng.random()).collect();
            let mut fused = original.clone();
            let mut paired = original.clone();
            pass::<F, false, false, 16>(&mut fused, width);
            layers::<F, false, false>(&mut paired, width, 0, 4);
            assert_eq!(fused, paired);

            pass::<F, true, false, 16>(&mut fused, width);
            assert_eq!(fused, original);
        }
    }

    fn check_transform(log_len: usize, log_width: usize, seed: u64) {
        let width = 1 << log_width;
        let mut rng = SmallRng::seed_from_u64(seed);
        let input = (0..width << log_len)
            .map(|_| rng.random::<F>())
            .collect::<Vec<_>>();
        let mut expected = input.clone();
        naive_forward(&mut expected, width);
        let mut actual = input.clone();
        forward_transform(&mut actual, width);
        assert_eq!(actual, expected);
        inverse_transform(&mut actual, width);
        naive_inverse(&mut expected, width);
        assert_eq!(actual, input);
        assert_eq!(actual, expected);
    }

    #[test]
    fn every_requested_shape() {
        for log_len in 1..=14 {
            for log_width in [0, 1, 2, 4, 5] {
                check_transform(log_len, log_width, 92);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn transform_matches_radix2(
            log_len in 1usize..=14,
            log_width in prop::sample::select(&[0usize, 1, 2, 4, 5][..]),
            seed in any::<u64>(),
        ) {
            check_transform(log_len, log_width, seed);
        }
    }
}
