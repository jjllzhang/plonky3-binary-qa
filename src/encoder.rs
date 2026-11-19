//! Systematic encoding with a cache-local forward/product/inverse sweep.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem;

use p3_field::{Algebra, ExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, transpose};

use crate::QaField;
use crate::kernel::{CACHE_BYTES, cache_log, encoding_cache_log, layers, upper_layers};

const ROW_TILE: usize = 16;
const PARITY_WORK_MIN_BYTES: usize = 64 * 1024 * 1024;

/// Systematic quadratic quotient ring encoder with fixed public spectra.
///
/// Coefficient and spectrum indices are both in natural order. Bit i of a
/// coefficient index is the exponent of the ith variable. Bit i of a spectrum
/// index selects either ω (0) or ω² (1) on that axis.
#[derive(Clone, Debug)]
pub struct QaEncoder<F, FMult = F> {
    log_message_len: usize,
    spectra: Vec<FMult>,
    field: PhantomData<F>,
}

impl<F: QaField + Algebra<M>, M: QaField> QaEncoder<F, M> {
    /// Use consecutive spectra of length 2^log_message_len, one per parity block.
    ///
    /// There must be 2^log_inv_rate − 1 such spectra. Use the forward transform
    /// to obtain them from coefficient vectors of ring elements.
    pub fn new(log_message_len: usize, log_inv_rate: usize, spectra: Vec<M>) -> Self {
        let (message_len, parity_count) = crate::geometry(log_message_len, log_inv_rate);
        assert_eq!(spectra.len(), message_len * parity_count);
        Self {
            log_message_len,
            spectra,
            field: PhantomData,
        }
    }

    /// Encode one coefficient vector, returning the message followed by its parity blocks.
    pub fn encode(&self, message: &[F]) -> Vec<F> {
        self.interleaved_encode_batch(message, 0)
    }

    /// Encode an interleaved batch with 2^log_batch_size messages.
    ///
    /// The input is ordered as [coordinate][message]. The output uses the same
    /// layout, with the original coordinates followed by each parity block.
    pub fn interleaved_encode_batch(&self, message: &[F], log_batch_size: usize) -> Vec<F> {
        if !self.spectra.is_empty()
            && size_of_val(message) >= PARITY_WORK_MIN_BYTES
            && current_num_threads() > 1
        {
            self.interleaved_using_last_parity(message, log_batch_size)
        } else {
            let mut output = Vec::new();
            self.interleaved_encode_batch_into(
                message,
                log_batch_size,
                &mut Vec::new(),
                &mut output,
            );
            output
        }
    }

    /// Encode an interleaved batch using caller-owned buffers.
    ///
    /// The buffers may be empty. The entire output is replaced with the codeword.
    /// Pass the buffers back on later calls to reuse their allocations.
    pub fn interleaved_encode_batch_into(
        &self,
        message: &[F],
        log_batch_size: usize,
        scratch: &mut Vec<F>,
        output: &mut Vec<F>,
    ) {
        self.interleaved(message, log_batch_size, scratch, output, F::scale_spectrum);
    }

    /// Encode interleaved subfield coefficients into the output field.
    ///
    /// The forward CRT stays in FSub. Systematic coefficients are embedded
    /// directly in the output and each leaf multiplies in mixed fields.
    pub fn subfield_encode_batch<FSub>(&self, message: &[FSub], log_batch_size: usize) -> Vec<F>
    where
        FSub: QaField,
        F: ExtensionField<FSub>,
    {
        let mut output = Vec::new();
        self.subfield_encode_batch_into(message, log_batch_size, &mut Vec::new(), &mut output);
        output
    }

    /// Encode interleaved subfield coefficients using caller-owned buffers.
    ///
    /// Scratch storage remains in the input field. The entire output is replaced
    /// with the codeword. The buffers may be empty or reused from a prior call.
    pub fn subfield_encode_batch_into<FSub>(
        &self,
        message: &[FSub],
        log_batch_size: usize,
        scratch: &mut Vec<FSub>,
        output: &mut Vec<F>,
    ) where
        FSub: QaField,
        F: ExtensionField<FSub>,
    {
        self.interleaved(
            message,
            log_batch_size,
            scratch,
            output,
            F::scale_subfield_spectrum,
        );
    }

    /// Encode rows of a matrix whose width is the message length.
    ///
    /// Each output row contains one codeword. Small groups of rows are
    /// transposed into interleaved scratch space so they can share the spectrum
    /// reads and use whole-register butterflies.
    pub fn encode_batch(&self, message: &RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let k = 1 << self.log_message_len;
        assert_eq!(message.width, k);
        let out_width = k + self.spectra.len();
        let height = message.values.len() / k;
        let mut output = F::zero_vec(height * out_width);
        let reuse = k >= 4096 && height >= 512 && current_num_threads() > 1;
        let row_tile = if reuse {
            if out_width >= 4 * k { 4 } else { 8 }
        } else {
            ROW_TILE
        };
        let write_tile = |out: &mut [F], input: &[F], scratch: &mut [F], parity: &mut [F]| {
            let width = input.len() / k;
            self.encode_parity::<F, false>(scratch, parity, width, F::scale_spectrum);

            for (row, source) in out.chunks_exact_mut(out_width).zip(input.chunks_exact(k)) {
                row[..k].copy_from_slice(source);
            }
            // Visit small coordinate strips so the strided reads stay in cache.
            for (strip, columns) in parity.chunks(32 * width).enumerate() {
                let start = k + 32 * strip;
                for (i, row) in out.chunks_exact_mut(out_width).enumerate() {
                    for (dst, column) in row[start..].iter_mut().zip(columns.chunks_exact(width)) {
                        *dst = column[i];
                    }
                }
            }
        };
        let tiles = output
            .par_chunks_mut(row_tile * out_width)
            .zip(message.values.par_chunks(row_tile * k));
        if reuse {
            tiles.with_min_len(4).for_each_init(
                || (Vec::new(), Vec::new()),
                |(scratch, parity), (out, input)| {
                    let width = input.len() / k;
                    resize_buffer(scratch, input.len());
                    resize_buffer(parity, width * self.spectra.len());
                    transpose::transpose(input, scratch, k, width);
                    write_tile(out, input, scratch, parity);
                },
            );
        } else {
            tiles.for_each(|(out, input)| {
                let width = input.len() / k;
                let mut scratch = transpose::transpose_owned(input, k, width);
                let mut parity = F::zero_vec(width * self.spectra.len());
                write_tile(out, input, &mut scratch, &mut parity);
            });
        }
        RowMajorMatrix::new(output, out_width)
    }

    /// Multiply a transformed batch by one of the fixed spectra.
    ///
    /// parity_index is zero-based. Both buffers must use the same interleaved
    /// layout and contain exactly 2^log_message_len coordinates.
    pub fn multiply_spectrum(
        &self,
        spectrum: &[F],
        width: usize,
        parity_index: usize,
        out: &mut [F],
    ) {
        let k = 1 << self.log_message_len;
        assert!(width != 0);
        assert_eq!(spectrum.len(), k * width);
        assert_eq!(out.len(), spectrum.len());
        let multipliers = &self.spectra[parity_index * k..(parity_index + 1) * k];
        let tile = 1 << cache_log::<F>(width, self.log_message_len);
        out.par_chunks_exact_mut(tile * width)
            .zip(spectrum.par_chunks_exact(tile * width))
            .zip(multipliers.par_chunks_exact(tile))
            .with_min_len(2)
            .for_each(|((dst, src), a)| F::scale_spectrum(src, a, dst, width));
    }

    fn interleaved<S: QaField>(
        &self,
        message: &[S],
        log_batch_size: usize,
        scratch: &mut Vec<S>,
        output: &mut Vec<F>,
        multiply: impl Fn(&[S], &[M], &mut [F], usize) + Sync,
    ) where
        F: Algebra<S>,
    {
        let log_len = log2_strict_usize(message.len());
        assert_eq!(
            log_len.checked_sub(self.log_message_len),
            Some(log_batch_size)
        );
        let width = 1 << log_batch_size;
        resize_buffer(output, message.len() + width * self.spectra.len());
        let (systematic, parity) = output.split_at_mut(message.len());
        let chunk = CACHE_BYTES / size_of::<F>();
        if parity.is_empty() {
            systematic
                .par_chunks_mut(chunk)
                .zip(message.par_chunks(chunk))
                .for_each(|(dst, src)| embed(src, dst));
        } else {
            resize_buffer(scratch, message.len());
            systematic
                .par_chunks_mut(chunk)
                .zip(scratch.par_chunks_mut(chunk))
                .zip(message.par_chunks(chunk))
                .for_each(|((systematic, work), src)| {
                    embed(src, systematic);
                    work.copy_from_slice(src);
                });
            self.encode_parity::<S, true>(scratch, parity, width, multiply);
        }
    }

    fn interleaved_using_last_parity(&self, message: &[F], log_batch_size: usize) -> Vec<F> {
        let log_len = log2_strict_usize(message.len());
        assert_eq!(
            log_len.checked_sub(self.log_message_len),
            Some(log_batch_size)
        );
        let width = 1 << log_batch_size;
        let message_len = message.len();
        let mut output = F::zero_vec(message_len + width * self.spectra.len());
        let (systematic, parity) = output.split_at_mut(message_len);
        // Keep the forward spectrum here until all earlier parity blocks use it.
        let (other_parity, work) = parity.split_at_mut(parity.len() - message_len);
        let chunk = CACHE_BYTES / size_of::<F>();
        systematic
            .par_chunks_mut(chunk)
            .zip(work.par_chunks_mut(chunk))
            .zip(message.par_chunks(chunk))
            .for_each(|((systematic, work), src)| {
                systematic.copy_from_slice(src);
                work.copy_from_slice(src);
            });

        let bottom = encoding_cache_log::<F, true>(width, self.log_message_len);
        let k = 1 << self.log_message_len;
        let last_spectrum = &self.spectra[self.spectra.len() - k..];
        self.encode_parity_and_then::<F, true>(
            work,
            other_parity,
            width,
            F::scale_spectrum,
            |leaf, first, local| {
                resize_buffer(local, leaf.len());
                let rows = leaf.len() / width;
                F::scale_spectrum(leaf, &last_spectrum[first..first + rows], local, width);
                layers::<F, true, false>(local, width, 0, bottom);
                leaf.copy_from_slice(local);
            },
        );
        upper_layers::<F, true, true>(work, width, bottom, self.log_message_len);
        output
    }

    fn encode_parity<S: QaField, const PARALLEL: bool>(
        &self,
        scratch: &mut [S],
        parity: &mut [F],
        width: usize,
        multiply: impl Fn(&[S], &[M], &mut [F], usize) + Sync,
    ) {
        if !parity.is_empty() {
            self.encode_parity_and_then::<S, PARALLEL>(
                scratch,
                parity,
                width,
                multiply,
                |_, _, _| {},
            );
        }
    }

    fn encode_parity_and_then<S: QaField, const PARALLEL: bool>(
        &self,
        scratch: &mut [S],
        parity: &mut [F],
        width: usize,
        multiply: impl Fn(&[S], &[M], &mut [F], usize) + Sync,
        finish_leaf: impl Fn(&mut [S], usize, &mut Vec<F>) + Sync,
    ) {
        let log_len = self.log_message_len;
        let bottom = encoding_cache_log::<F, PARALLEL>(width, log_len);
        upper_layers::<S, false, PARALLEL>(scratch, width, bottom, log_len);

        let mut blocks = parity.chunks_exact_mut(scratch.len()).collect::<Vec<_>>();
        let fuse = |subtree: &mut [S], blocks: &mut [&mut [F]], first: usize| {
            let mut local = Vec::new();
            let leaf_rows = 1 << bottom;
            let leaf_len = width * leaf_rows;
            for (leaf_index, leaf) in subtree.chunks_exact_mut(leaf_len).enumerate() {
                layers::<S, false, false>(leaf, width, 0, bottom);
                let offset = leaf_index * leaf_len;
                let first_row = first + leaf_index * leaf_rows;
                for (out, spectrum) in blocks
                    .iter_mut()
                    .zip(self.spectra.chunks_exact(1 << log_len))
                {
                    let out = &mut out[offset..offset + leaf_len];
                    multiply(
                        leaf,
                        &spectrum[first_row..first_row + leaf_rows],
                        out,
                        width,
                    );
                    layers::<F, true, false>(out, width, 0, bottom);
                }
                finish_leaf(leaf, first_row, &mut local);
            }
        };
        let block_len = scratch.len();
        if PARALLEL {
            let leaves = block_len / (width << bottom);
            let tasks = current_num_threads().saturating_mul(4).min(leaves);
            split_leaves(scratch, &mut blocks, width, 0, tasks, &fuse);
        } else {
            fuse(scratch, &mut blocks, 0);
        }
        // The split narrows its mutable views. Reborrow each complete parity block
        // for the upper inverse layers, which couple coordinates across leaves.
        for block in parity.chunks_exact_mut(block_len) {
            upper_layers::<F, true, PARALLEL>(block, width, bottom, log_len);
        }
    }
}

/// Encode a coordinate-interleaved batch with the fixed spectra of an encoder.
pub fn interleaved_encode_batch<F, M>(
    encoder: &QaEncoder<F, M>,
    message: &[F],
    log_batch_size: usize,
) -> Vec<F>
where
    F: QaField + Algebra<M>,
    M: QaField,
{
    encoder.interleaved_encode_batch(message, log_batch_size)
}

/// Encode subfield coefficients from a coordinate-interleaved batch.
pub fn subfield_encode_batch<FSub, F, M>(
    encoder: &QaEncoder<F, M>,
    message: &[FSub],
    log_batch_size: usize,
) -> Vec<F>
where
    FSub: QaField,
    F: QaField + Algebra<M> + ExtensionField<FSub>,
    M: QaField,
{
    encoder.subfield_encode_batch(message, log_batch_size)
}

fn resize_buffer<F: QaField>(buffer: &mut Vec<F>, len: usize) {
    if buffer.capacity() < len {
        *buffer = F::zero_vec(len);
    } else {
        buffer.resize(len, F::ZERO);
    }
}

/// Split all parity blocks at the same coordinate, retaining the final layout.
fn split_leaves<F: QaField, S: QaField>(
    scratch: &mut [S],
    parity: &mut [&mut [F]],
    width: usize,
    first: usize,
    tasks: usize,
    fuse: &(impl Fn(&mut [S], &mut [&mut [F]], usize) + Sync),
) {
    if tasks <= 1 {
        fuse(scratch, parity, first);
        return;
    }

    let half = scratch.len() / 2;
    let (left, right) = scratch.split_at_mut(half);
    let mut right_parity = Vec::with_capacity(parity.len());
    for part in parity.iter_mut() {
        let (head, tail) = mem::take(part).split_at_mut(half);
        *part = head;
        right_parity.push(tail);
    }
    let tasks = tasks.div_ceil(2);
    join(
        || split_leaves(left, parity, width, first, tasks, fuse),
        || {
            split_leaves(
                right,
                &mut right_parity,
                width,
                first + half / width,
                tasks,
                fuse,
            );
        },
    );
}

#[inline]
fn embed<S: QaField, F: QaField + Algebra<S>>(input: &[S], out: &mut [F]) {
    for (dst, &src) in out.iter_mut().zip(input) {
        *dst = F::from(src);
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128 as F, TowerLevel,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::{NaiveQaEncoder, forward_transform, inverse_transform};

    #[test]
    fn hand_computed_x_multiplier() {
        let (x, y) = (F::from_repr(137), F::from_repr(299));
        let a = F::omega();
        let encoder = QaEncoder::<F>::new(1, 1, alloc::vec![a, a.square()]);
        assert_eq!(encoder.encode(&[x, y]), [x, y, y, x + y]);
    }

    fn random<T>(rng: &mut SmallRng, len: usize) -> Vec<T>
    where
        StandardUniform: Distribution<T>,
    {
        (0..len).map(|_| rng.random()).collect()
    }

    #[test]
    fn all_batch_layouts_match_ring_products() {
        let mut rng = SmallRng::seed_from_u64(123);
        for s in 1..=6 {
            for rate in [1, 2] {
                let k = 1 << s;
                let width = 32;
                let multipliers = random::<F>(&mut rng, ((1 << rate) - 1) * k);
                let naive = NaiveQaEncoder::<F>::new(s, rate, &multipliers);
                let encoder = QaEncoder::<F>::new(s, rate, multipliers);
                let message = RowMajorMatrix::new(random::<F>(&mut rng, width * k), k);
                let interleaved = message.transpose();
                let expected = RowMajorMatrix::new(
                    message
                        .values
                        .chunks_exact(k)
                        .flat_map(|v| naive.encode(v))
                        .collect(),
                    k << rate,
                );

                assert_eq!(
                    encoder.encode(&message.values[..k]),
                    expected.values[..k << rate]
                );
                assert_eq!(encoder.encode_batch(&message), expected);
                assert_eq!(
                    encoder.interleaved_encode_batch(&interleaved.values, 5),
                    expected.transpose().values,
                );
            }
        }
    }

    fn check_subfield<S>(rng: &mut SmallRng)
    where
        S: QaField,
        F: ExtensionField<S>,
        StandardUniform: Distribution<S>,
    {
        for s in 1..=6 {
            let k = 1 << s;
            let width = 8;
            let spectra = random::<F>(rng, 3 * k);
            let messages = random::<S>(rng, width * k);
            let naive = NaiveQaEncoder::<F>::new(s, 2, &spectra);
            let encoder = QaEncoder::<F>::new(s, 2, spectra);
            let wide = messages.iter().copied().map(F::from).collect();
            let expected = RowMajorMatrix::new(wide, width).transpose();
            let expected = RowMajorMatrix::new(
                expected
                    .values
                    .chunks_exact(k)
                    .flat_map(|v| naive.encode(v))
                    .collect(),
                k * 4,
            )
            .transpose();

            assert_eq!(encoder.subfield_encode_batch(&messages, 3), expected.values);
        }
    }

    #[test]
    fn interleaved_buffers_can_be_reused_with_a_different_batch_width() {
        let mut rng = SmallRng::seed_from_u64(41);
        let log_k = 5;
        let encoder = QaEncoder::<F>::new(log_k, 2, random::<F>(&mut rng, 3 << log_k));
        let mut scratch = Vec::new();
        let mut narrow_scratch = Vec::new();
        let mut output = Vec::new();
        for log_width in [3, 5, 1, 4] {
            let input = random(&mut rng, 1 << (log_k + log_width));
            encoder.interleaved_encode_batch_into(&input, log_width, &mut scratch, &mut output);
            assert_eq!(output, encoder.interleaved_encode_batch(&input, log_width));

            let input = random::<BinaryField16>(&mut rng, 1 << (log_k + log_width));
            encoder.subfield_encode_batch_into(&input, log_width, &mut narrow_scratch, &mut output);
            assert_eq!(output, encoder.subfield_encode_batch(&input, log_width));
        }
    }

    #[test]
    fn last_parity_workspace_matches_separate_scratch() {
        let mut rng = SmallRng::seed_from_u64(222);
        for (log_k, log_width) in [(3, 2), (9, 5)] {
            let input = random::<F>(&mut rng, 1 << (log_k + log_width));
            for log_c in [1, 2] {
                let spectra = random::<F>(&mut rng, ((1 << log_c) - 1) << log_k);
                let encoder = QaEncoder::new(log_k, log_c, spectra);
                assert_eq!(
                    encoder.interleaved_using_last_parity(&input, log_width),
                    encoder.interleaved_encode_batch(&input, log_width),
                );
            }
        }
    }

    #[test]
    fn narrow_witnesses_match_ring_products() {
        let mut rng = SmallRng::seed_from_u64(444);
        check_subfield::<BinaryField8>(&mut rng);
        check_subfield::<BinaryField16>(&mut rng);
        check_subfield::<BinaryField32>(&mut rng);
        check_subfield::<BinaryField64>(&mut rng);
    }

    #[test]
    fn narrow_multipliers_and_partial_row_tile_match_ring_products() {
        let mut rng = SmallRng::seed_from_u64(5);
        for s in 1..=6 {
            let k = 1 << s;
            let spectra = random::<BinaryField64>(&mut rng, 3 * k);
            let naive = NaiveQaEncoder::<F>::new(s, 2, &spectra);
            let encoder = QaEncoder::<F, BinaryField64>::new(s, 2, spectra);
            let input = RowMajorMatrix::new(random::<F>(&mut rng, 19 * k), k);
            let actual = encoder.encode_batch(&input);
            assert_eq!(actual.height(), 19);
            for (input, actual) in input
                .values
                .chunks_exact(k)
                .zip(actual.values.chunks_exact(4 * k))
            {
                assert_eq!(actual, naive.encode(input));
            }
        }
    }

    #[test]
    fn fused_batches_match_separate_stages_across_cache_leaves() {
        let mut rng = SmallRng::seed_from_u64(88);
        let s = 12;
        let width = 32;
        let k = 1 << s;
        let len = k * width;
        let spectra = random::<F>(&mut rng, 3 * k);
        let message = random::<F>(&mut rng, len);
        let encoder = QaEncoder::<F>::new(s, 2, spectra);
        let fused = encoder.interleaved_encode_batch(&message, 5);

        let mut spectrum = message.clone();
        forward_transform(&mut spectrum, width);
        let mut parity = F::zero_vec(len);
        assert_eq!(fused[..len], message);
        for i in 0..3 {
            encoder.multiply_spectrum(&spectrum, width, i, &mut parity);
            inverse_transform(&mut parity, width);
            let block = &fused[(i + 1) * len..(i + 2) * len];
            let mismatch = block.iter().zip(&parity).position(|(a, b)| a != b);
            assert_eq!(mismatch, None, "parity block {i}");
        }
    }
}
