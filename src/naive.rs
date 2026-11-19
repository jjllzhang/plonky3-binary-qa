//! Scalar reference for the tensor transform and the quotient-ring product.

use alloc::vec::Vec;

use p3_field::Algebra;

use crate::QaField;

/// Apply the CRT evaluation in natural order with ordinary field multiplication.
///
/// `width` is the number of interleaved messages. Each coordinate occupies
/// `width` consecutive elements.
pub fn naive_forward<F: QaField>(values: &mut [F], width: usize) {
    radix2::<F, false>(values, width);
}

/// Interpolate the CRT evaluations using ordinary field multiplication.
pub fn naive_inverse<F: QaField>(values: &mut [F], width: usize) {
    radix2::<F, true>(values, width);
}

fn radix2<F: QaField, const INVERSE: bool>(values: &mut [F], width: usize) {
    assert!(width != 0 && values.len().is_multiple_of(width));
    assert!((values.len() / width).is_power_of_two());
    let omega = F::omega();
    let omega_squared = omega.square();

    let mut stride = width;
    while stride < values.len() {
        for block in values.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            for (x, y) in lo.iter_mut().zip(hi) {
                (*x, *y) = if INVERSE {
                    (omega_squared * *x + omega * *y, *x + *y)
                } else {
                    (*x + omega * *y, *x + omega_squared * *y)
                };
            }
        }
        stride *= 2;
    }
}

/// Schoolbook product modulo `X_i² + X_i + 1` for every variable.
///
/// The coefficient at index `b` belongs to the monomial with exponents
/// given by the bits of `b`. Both inputs must have the same power-of-two
/// length. With `K` coefficients this uses `K²` field products.
pub fn quotient_product<F: QaField>(lhs: &[F], rhs: &[F]) -> Vec<F> {
    assert!(lhs.len().is_power_of_two());
    assert_eq!(lhs.len(), rhs.len());
    let mut product = F::zero_vec(lhs.len());
    let mut scratch = F::zero_vec(lhs.len() - 1);
    add_product(lhs, rhs, &mut product, &mut scratch);
    product
}

fn add_product<F: QaField>(lhs: &[F], rhs: &[F], out: &mut [F], scratch: &mut [F]) {
    if lhs.len() == 1 {
        out[0] += lhs[0] * rhs[0];
        return;
    }

    // (a + bX)(c + dX) = ac + bd + (ad + bc + bd)X in this quotient.
    let half = lhs.len() / 2;
    let (a, b) = lhs.split_at(half);
    let (c, d) = rhs.split_at(half);
    let (lo, hi) = out.split_at_mut(half);
    let (bd, inner) = scratch.split_at_mut(half);
    add_product(a, c, lo, inner);
    add_product(a, d, hi, inner);
    add_product(b, c, hi, inner);
    bd.fill(F::ZERO);
    add_product(b, d, bd, inner);
    for ((lo, hi), &square_product) in lo.iter_mut().zip(hi).zip(bd.iter()) {
        *lo += square_product;
        *hi += square_product;
    }
}

/// Systematic QA encoder based on direct polynomial multiplication.
#[derive(Clone, Debug)]
pub struct NaiveQaEncoder<F> {
    message_len: usize,
    multipliers: Vec<F>,
}

impl<F: QaField> NaiveQaEncoder<F> {
    /// Recover the coefficient multipliers from consecutive spectra of length
    /// `2^log_message_len`, one spectrum per parity block.
    pub fn new<M>(log_message_len: usize, log_inv_rate: usize, spectra: &[M]) -> Self
    where
        M: QaField,
        F: Algebra<M>,
    {
        let (message_len, parity_count) = crate::geometry(log_message_len, log_inv_rate);
        assert_eq!(spectra.len(), message_len * parity_count);
        let mut multipliers = spectra.iter().copied().map(F::from).collect::<Vec<_>>();
        for coeffs in multipliers.chunks_exact_mut(message_len) {
            naive_inverse(coeffs, 1);
        }
        Self {
            message_len,
            multipliers,
        }
    }

    /// Append the quotient-ring products to the original coefficient vector.
    pub fn encode(&self, message: &[F]) -> Vec<F> {
        assert_eq!(message.len(), self.message_len);
        let mut codeword = Vec::with_capacity(message.len() + self.multipliers.len());
        codeword.extend_from_slice(message);
        for multiplier in self.multipliers.chunks_exact(self.message_len) {
            codeword.extend(quotient_product(message, multiplier));
        }
        codeword
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8 as F, TowerLevel};
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn multiplication_by_x() {
        let (x, y) = (F::from_repr(37), F::from_repr(99));
        assert_eq!(quotient_product(&[x, y], &[F::ZERO, F::ONE]), [y, x + y]);
        let encoder = NaiveQaEncoder::<F>::new(1, 1, &[F::omega(), F::omega().square()]);
        assert_eq!(encoder.encode(&[x, y]), [x, y, y, x + y]);
    }
}
