//! Pointwise products, with one spectral multiplier shared by a row of witnesses.

use p3_binary_field::{BinaryField64, BinaryField128 as F128, Poly64, TowerLevel, poly_basis};
use p3_field::{Algebra, ExtensionField, Field, PackedValue};

use crate::QaField;

type Packed64 = <Poly64 as Field>::Packing;
const LANES: usize = Packed64::WIDTH;

#[inline]
pub(crate) fn scale_rows<F, M>(input: &[F], multipliers: &[M], out: &mut [F], width: usize)
where
    F: QaField + Algebra<M>,
    M: QaField,
{
    for ((src, dst), &a) in input
        .chunks_exact(width)
        .zip(out.chunks_exact_mut(width))
        .zip(multipliers)
    {
        for (&src, dst) in src.iter().zip(dst) {
            *dst = src * a;
        }
    }
}

#[inline]
pub(crate) fn scale_subfield_rows<S, F, M>(
    input: &[S],
    multipliers: &[M],
    out: &mut [F],
    width: usize,
) where
    S: QaField,
    F: QaField + Algebra<M> + ExtensionField<S>,
    M: QaField,
{
    for ((src, dst), &a) in input
        .chunks_exact(width)
        .zip(out.chunks_exact_mut(width))
        .zip(multipliers)
    {
        let a = F::from(a);
        for (&src, dst) in src.iter().zip(dst) {
            *dst = a * src;
        }
    }
}

#[inline]
pub(crate) fn scale_bf128<M>(input: &[F128], multipliers: &[M], out: &mut [F128], width: usize)
where
    M: QaField,
    F128: Algebra<M>,
{
    if poly_basis::HAS_HARDWARE_CLMUL && size_of::<M>() == size_of::<F128>() && width >= 4 {
        // Convert whole cache leaves while both CRT transforms use the tower basis.
        out.copy_from_slice(input);
        let out = F128::as_repr_slice_mut(out);
        poly_basis::from_tower_slice(out);
        for (column, &a) in out.chunks_exact_mut(width).zip(multipliers) {
            poly_basis::mul_slice(column, poly_basis::from_tower(F128::from(a)));
        }
        poly_basis::to_tower_slice(out);
    } else if LANES > 1 && size_of::<M>() == size_of::<BinaryField64>() && width >= LANES {
        scale_bf64_multipliers(input, multipliers, out, width);
    } else {
        scale_rows(input, multipliers, out, width);
    }
}

/// A BF128 element has two coordinates over BF64. Multiplication by a
/// subfield element scales both independently. Use Plonky3's packed Poly64
/// for those two BF64 products, then reassemble the tower coordinates.
#[inline]
pub(crate) fn scale_bf128_subfield<S, M>(
    input: &[S],
    multipliers: &[M],
    out: &mut [F128],
    width: usize,
) where
    S: QaField,
    M: QaField,
    F128: Algebra<M> + ExtensionField<S>,
{
    if LANES == 1 || size_of::<S>() > size_of::<BinaryField64>() || width < LANES {
        scale_subfield_rows(input, multipliers, out, width);
        return;
    }

    for ((src, dst), &a) in input
        .chunks_exact(width)
        .zip(out.chunks_exact_mut(width))
        .zip(multipliers)
    {
        let a = F128::from(a);
        let (a0, a1) = split(a);
        let mut src = src.chunks_exact(LANES);
        let mut dst = dst.chunks_exact_mut(LANES);
        for (src, dst) in src.by_ref().zip(dst.by_ref()) {
            // The witness embedding into BF64 is the original tower bit pattern.
            let x = Packed64::from_fn(|i| poly64(F128::from(src[i]).to_repr() as u64));
            join_products(dst, x * a0, x * a1);
        }
        for (&x, dst) in src.remainder().iter().zip(dst.into_remainder()) {
            *dst = a * x;
        }
    }
}

#[inline]
fn scale_bf64_multipliers<M: QaField>(
    input: &[F128],
    multipliers: &[M],
    out: &mut [F128],
    width: usize,
) where
    F128: Algebra<M>,
{
    for ((src, dst), &a) in input
        .chunks_exact(width)
        .zip(out.chunks_exact_mut(width))
        .zip(multipliers)
    {
        let scale = poly64(F128::from(a).to_repr() as u64);
        let mut src = src.chunks_exact(LANES);
        let mut dst = dst.chunks_exact_mut(LANES);
        for (src, dst) in src.by_ref().zip(dst.by_ref()) {
            let lo = Packed64::from_fn(|i| poly64(src[i].to_repr() as u64));
            let hi = Packed64::from_fn(|i| poly64((src[i].to_repr() >> 64) as u64));
            join_products(dst, lo * scale, hi * scale);
        }
        for (&x, dst) in src.remainder().iter().zip(dst.into_remainder()) {
            *dst = x * a;
        }
    }
}

#[inline]
fn poly64(bits: u64) -> Poly64 {
    Poly64::from(BinaryField64::from_repr(bits))
}

#[inline]
fn split(x: F128) -> (Poly64, Poly64) {
    let bits = x.to_repr();
    (poly64(bits as u64), poly64((bits >> 64) as u64))
}

#[inline]
fn join_products(out: &mut [F128], lo: Packed64, hi: Packed64) {
    for ((dst, &lo), &hi) in out.iter_mut().zip(lo.as_slice()).zip(hi.as_slice()) {
        let lo = u128::from(BinaryField64::from(lo).to_repr());
        let hi = u128::from(BinaryField64::from(hi).to_repr());
        *dst = F128::from_repr(lo | (hi << 64));
    }
}
