//! Tower-basis arithmetic for the tensor butterflies.

use p3_binary_field::{
    BinaryField2, BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, M128,
    M256, M512, PackedGf2, TowerLevel,
};
use p3_field::{Algebra, ExtensionField};

use crate::product::{scale_bf128, scale_bf128_subfield, scale_rows, scale_subfield_rows};

/// A scalar or packed tower value on which the CRT butterflies operate lane-wise.
pub trait QaButterflyValue: Copy + Send + Sync {
    /// Multiply every lane by the tower embedding of the root `ω` of `X² + X + 1`.
    fn mul_omega(self) -> Self;

    /// Add in characteristic two.
    fn bitxor(self, other: Self) -> Self;

    /// Replace `(x, y)` with `(x + ωy, x + ω²y)`.
    #[inline(always)]
    fn butterfly_t(x: &mut Self, y: &mut Self) {
        *x = x.bitxor(y.mul_omega());
        *y = x.bitxor(*y);
    }

    /// Undo the forward butterfly.
    #[inline(always)]
    fn butterfly_u(x: &mut Self, y: &mut Self) {
        *y = x.bitxor(*y);
        *x = x.bitxor(y.mul_omega());
    }

    /// Apply two adjacent forward layers.
    #[inline(always)]
    fn butterfly_t_radix4(x0: &mut Self, x1: &mut Self, x2: &mut Self, x3: &mut Self) {
        Self::butterfly_t(x0, x2);
        Self::butterfly_t(x1, x3);
        Self::butterfly_t(x0, x1);
        Self::butterfly_t(x2, x3);
    }

    /// Apply two adjacent inverse layers.
    #[inline(always)]
    fn butterfly_u_radix4(x0: &mut Self, x1: &mut Self, x2: &mut Self, x3: &mut Self) {
        Self::butterfly_u(x0, x1);
        Self::butterfly_u(x2, x3);
        Self::butterfly_u(x0, x2);
        Self::butterfly_u(x1, x3);
    }
}

/// A byte-aligned Wiedemann tower field supported by the QA encoder.
///
/// Its Plonky3 `Field::Packing` is the scalar itself. The CRT kernels process
/// additional packed tower words across independent butterflies.
pub trait QaField: TowerLevel<Packing = Self> + QaButterflyValue + private::Sealed {
    /// The image of the two-bit tower generator.
    fn omega() -> Self;

    #[doc(hidden)]
    fn scale_spectrum<M: QaField>(input: &[Self], multipliers: &[M], out: &mut [Self], width: usize)
    where
        Self: Algebra<M>,
    {
        scale_rows(input, multipliers, out, width);
    }
    #[doc(hidden)]
    fn scale_subfield_spectrum<S: QaField, M: QaField>(
        input: &[S],
        multipliers: &[M],
        out: &mut [Self],
        width: usize,
    ) where
        Self: Algebra<M> + ExtensionField<S>,
    {
        scale_subfield_rows(input, multipliers, out, width);
    }
}

mod private {
    pub trait Sealed {}
}

// Every narrower tower is in the low bits of BF128. Each GF(4) coordinate
// is scaled independently, so the same Plonky3 operation also scales one
// packed 64-bit word without mixing its adjacent tower elements.
#[inline(always)]
fn mul_omega_bits(bits: u128) -> u128 {
    (BinaryField128::from_repr(bits) * BinaryField2::from_repr(2)).to_repr()
}

macro_rules! impl_tower {
    ($field:ty => $word:ty $(, $scale:item)*) => {
        impl private::Sealed for $field {}

        impl QaField for $field {
            fn omega() -> Self { Self::from_repr(2) }
            $($scale)*
        }

        impl QaButterflyValue for $field {
            #[inline(always)]
            fn mul_omega(self) -> Self {
                Self::from_repr(mul_omega_bits(u128::from(self.to_repr())) as $word)
            }

            #[inline(always)]
            fn bitxor(self, other: Self) -> Self {
                self + other
            }
        }
    };
}

impl_tower!(BinaryField8 => u8);
impl_tower!(BinaryField16 => u16);
impl_tower!(BinaryField32 => u32);
impl_tower!(BinaryField64 => u64);
impl_tower!(
    BinaryField128 => u128,
    fn scale_spectrum<M: QaField>(input: &[Self], multipliers: &[M], out: &mut [Self], width: usize)
    where
        Self: Algebra<M>,
    {
        scale_bf128(input, multipliers, out, width);
    },
    fn scale_subfield_spectrum<S: QaField, M: QaField>(
        input: &[S],
        multipliers: &[M],
        out: &mut [Self],
        width: usize,
    ) where
        Self: Algebra<M> + ExtensionField<S>,
    {
        scale_bf128_subfield(input, multipliers, out, width);
    }
);

// The aligned word blocks let LLVM choose SIMD instructions for the target.
// Each two-bit coefficient is contained in a single u64, including at word boundaries.
macro_rules! impl_packed {
    ($($packed:ty),+ $(,)?) => {$(
        impl QaButterflyValue for $packed {
            #[inline(always)]
            fn mul_omega(self) -> Self {
                Self::from_words(self.to_words().map(|word| mul_omega_bits(u128::from(word)) as u64))
            }

            #[inline(always)]
            fn bitxor(self, other: Self) -> Self {
                (PackedGf2::new(self) + PackedGf2::new(other)).to_bits()
            }
        }
    )+};
}

impl_packed!(M128, M256, M512);

#[cfg(test)]
mod tests {
    use p3_binary_field::Underlier;
    use p3_field::{Field, PackedValue};
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn check_tower<F: QaField>()
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(42);
        assert_eq!(F::omega().square() + F::omega() + F::ONE, F::ZERO);
        for _ in 0..1024 {
            let x: F = rng.random();
            let ox = x.mul_omega();
            assert_eq!(ox, x * F::omega());
            assert_eq!(ox.mul_omega() + ox + x, F::ZERO);
            let packed = <F as Field>::Packing::from(x);
            assert_eq!(packed.mul_omega().as_slice(), &[x * F::omega()]);
        }
    }

    #[test]
    fn omega_matches_tower_multiplication() {
        check_tower::<BinaryField8>();
        check_tower::<BinaryField16>();
        check_tower::<BinaryField32>();
        check_tower::<BinaryField64>();
        check_tower::<BinaryField128>();
    }

    fn check_packed<P: Underlier<Word = u64> + QaButterflyValue>() {
        let mut rng = SmallRng::seed_from_u64(43);
        for _ in 0..1024 {
            let x = P::random(&mut rng);
            let ox = x.mul_omega();
            for (&a, &b) in x.words().iter().zip(ox.words()) {
                assert_eq!(
                    b,
                    (BinaryField64::from_repr(a) * BinaryField64::omega()).to_repr()
                );
            }
            assert_eq!(ox.mul_omega().bitxor(ox).bitxor(x), P::ZERO);
        }
    }

    #[test]
    fn omega_matches_each_packed_word() {
        check_packed::<M128>();
        check_packed::<M256>();
        check_packed::<M512>();
    }
}
