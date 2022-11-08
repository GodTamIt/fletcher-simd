#![doc = include_str!("../README.md")]
#![feature(portable_simd)]
#![feature(stdsimd)]

use {
    core::{
        cmp::PartialEq,
        convert::{From, TryFrom},
        fmt::Debug,
        ops::{Add, AddAssign, IndexMut, Mul},
        simd::{LaneCount, Simd, SimdElement, SimdUint, SupportedLaneCount},
    },
    multiversion::multiversion,
    num::traits::{Num, Unsigned, WrappingAdd, WrappingMul, WrappingSub},
};

#[cfg(feature = "runtime_dispatch")]
#[allow(unused_imports)]
use std::arch::{is_aarch64_feature_detected, is_arm_feature_detected};

/// Trait for the type representing a certain sized Fletcher checksum.
pub trait FletcherChecksum: Num + Unsigned + Default {
    type BlockType: Copy
        + Clone
        + Default
        + PartialEq
        + SimdElement
        + TryFrom<usize>
        + Unsigned
        + WrappingAdd
        + WrappingSub;
}

/// A Fletcher checksum object that allows for continuous updates to the checksum.
///
/// # Examples
///
/// ```
/// use fletcher_simd::Fletcher;
///
/// const DATA: &str = "abcdefgh";
/// let mut fletcher = Fletcher::<u16>::new();
/// fletcher.update_with_slice(DATA.as_bytes());
///
/// assert_eq!(fletcher.value(), 0xF824);
/// ```
///
/// The [`update_with_iter`](Fletcher::update_with_iter) method is also available for use with the
/// [`Iterator`] interface.
///
/// ```
/// use byteorder::{ByteOrder, LittleEndian};
/// use fletcher_simd::Fletcher128;
///
/// const DATA: &str = "abcdefgh";
/// let mut fletcher = Fletcher128::new();
///
/// fletcher.update_with_iter(
///     DATA.as_bytes()
///         .chunks(8)
///         .map(|chunk| LittleEndian::read_u64(chunk)),
/// );
///
/// assert_eq!(fletcher.value(), 0x68676665646362616867666564636261);
/// ```
///
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Fletcher<T: FletcherChecksum> {
    a: T::BlockType,
    b: T::BlockType,
}

/// Currently, limit vector sizes to 256 bits. In the future, this may bump up to 512 bits for
/// AVX-512.
const MAX_VEC_SIZE: usize = 256 / 8;

/// Macro to implement [`Fletcher`] since the SIMD interface does not play well with inherent
/// associated types and outside generics.
macro_rules! impl_fletcher {
    ($result_type:ty, $block_type:ty, $block_size:literal) => {
        impl FletcherChecksum for $result_type {
            type BlockType = $block_type;
        }

        impl Fletcher<$result_type> {
            /// Constructs a new `Fletcher<T>` with the default values.
            pub fn new() -> Self {
                Self::default()
            }

            /// Constructs a new `Fletcher<T>` with specific values.
            ///
            /// `a` will represent the lesser significant bits.
            /// `b` will represent the more significant bits.
            pub fn with_initial_values(a: $block_type, b: $block_type) -> Self {
                Self { a, b }
            }

            /// Updates the checksum with a slice of data of type `T::BlockType`.
            pub fn update_with_slice(&mut self, data: &[$block_type]) {
                if data.is_empty() {
                    return;
                }

                const NUM_LANES: usize = MAX_VEC_SIZE / $block_size;

                let (simd_slice, remainder_slice) =
                    data.split_at(data.len() - (data.len() % NUM_LANES));

                if !simd_slice.is_empty() {
                    (self.a, self.b) = update_fletcher_simd(
                        self.a,
                        self.b,
                        simd_slice
                            .chunks(NUM_LANES)
                            .map(|slice| Simd::<$block_type, NUM_LANES>::from_slice(slice)),
                    );
                }

                if !remainder_slice.is_empty() {
                    (self.a, self.b) =
                        update_fletcher_scalar(self.a, self.b, remainder_slice.iter().copied());
                }
            }

            /// Updates the checksum with an iterator over elements of type `T::BlockType`.
            pub fn update_with_iter<Iter>(&mut self, elems: Iter)
            where
                Iter: Iterator<Item = $block_type>,
            {
                const NUM_LANES: usize = MAX_VEC_SIZE / $block_size;

                let mut simd_vec = Simd::<$block_type, NUM_LANES>::default();
                let mut simd_size = 0;

                // Grab chunks of `NUM_LANES` and feed them into the SIMD calculation.
                (self.a, self.b) = update_fletcher_simd(
                    self.a,
                    self.b,
                    elems.filter_map(|elem| {
                        simd_vec[simd_size] = elem;
                        simd_size += 1;

                        if simd_size == NUM_LANES {
                            simd_size = 0;
                            Some(simd_vec)
                        } else {
                            None
                        }
                    }),
                );

                // If the number elements are not a multiple of `NUM_LANES`, use scalar fallback to
                // compute remainder slice.
                if simd_size > 0 {
                    (self.a, self.b) = update_fletcher_scalar(
                        self.a,
                        self.b,
                        (0..simd_size).map(|idx| simd_vec[idx]),
                    );
                }
            }

            /// Updates the checksum with an iterator over elements of type `T::BlockType` using
            /// a scalar-only implementation.
            pub fn update_with_iter_scalar<Iter>(&mut self, elems: Iter)
            where
                Iter: Iterator<Item = $block_type>,
            {
                (self.a, self.b) = update_fletcher_scalar(self.a, self.b, elems);
            }

            /// Returns the checksum value.
            pub fn value(&self) -> $result_type {
                const SHIFT_SIZE: usize = core::mem::size_of::<$block_type>() * 8;

                ((self.b as $result_type) << SHIFT_SIZE) | self.a as $result_type
            }
        }

        impl From<Fletcher<$result_type>> for $result_type {
            fn from(f: Fletcher<$result_type>) -> Self {
                f.value()
            }
        }
    };
}

impl_fletcher!(u16, u8, 1);
impl_fletcher!(u32, u16, 2);
impl_fletcher!(u64, u32, 4);
impl_fletcher!(u128, u64, 8);

/// Convenient type alias for the 16-bit Fletcher checksum object.
pub type Fletcher16 = Fletcher<u16>;

/// Convenient type alias for the 32-bit Fletcher checksum object.
pub type Fletcher32 = Fletcher<u32>;

/// Convenient type alias for the 64-bit Fletcher checksum object.
pub type Fletcher64 = Fletcher<u64>;

/// Convenient type alias for the 128-bit Fletcher checksum object.
pub type Fletcher128 = Fletcher<u128>;

/// Private helper trait for making [`update_fletcher_simd`] generic.
trait FletcherSimdVec<T, const LANES: usize>:
    Add<Self, Output = Self>
    + AddAssign<Simd<T, LANES>>
    + AsRef<[T; LANES]>
    + Copy
    + Clone
    + Default
    + Mul<Self, Output = Self>
    + Sized
    + SimdUint
where
    T: Copy + Clone + Default + SimdElement + WrappingAdd + WrappingSub,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn horizontal_sum(self) -> T;
}

macro_rules! impl_simdvec {
    ($t:ty) => {
        impl<const LANES: usize> FletcherSimdVec<$t, LANES> for Simd<$t, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            #[inline]
            fn horizontal_sum(self) -> $t {
                self.reduce_sum()
            }
        }
    };
}

impl_simdvec!(u8);
impl_simdvec!(u16);
impl_simdvec!(u32);
impl_simdvec!(u64);

/// Function that updates a fletcher checksum using SIMD.
#[multiversion]
#[clone(target = "[x86|x86_64]+avx+avx2")]
#[clone(target = "[x86|x86_64]+avx")]
#[clone(target = "[x86|x86_64]+sse+sse2")]
#[clone(target = "[x86|x86_64]+sse")]
#[clone(target = "[arm|aarch64]+neon")]
fn update_fletcher_simd<BlockType, Iter, SimdVec, const LANES: usize>(
    mut a: BlockType,
    mut b: BlockType,
    elems: Iter,
) -> (BlockType, BlockType)
where
    BlockType: Copy
        + Clone
        + Default
        + TryFrom<usize>
        + SimdElement
        + Unsigned
        + WrappingAdd
        + WrappingMul
        + WrappingSub,
    <BlockType as TryFrom<usize>>::Error: Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Iter: Iterator<Item = SimdVec>,
    SimdVec: FletcherSimdVec<BlockType, LANES> + IndexMut<usize, Output = BlockType>,
{
    let mut a_accum = SimdVec::default();
    let mut b_accum = SimdVec::default();

    for elem in elems {
        a_accum = a_accum + elem;
        b_accum = b_accum + a_accum;
    }

    a = a.wrapping_add(&a_accum.horizontal_sum());

    // b += (LANES * b_accum)
    let b_prime = BlockType::wrapping_mul(
        &BlockType::try_from(LANES).unwrap(),
        &b_accum.horizontal_sum(),
    );
    b = b.wrapping_add(&b_prime);

    // b -= (i * a_accum[i]) for i in 0..LANES
    let increasing_mask = {
        let mut vec = SimdVec::default();
        (0..LANES).for_each(|i| vec[i] = BlockType::try_from(i).unwrap());
        vec
    };
    let sub_a = a_accum * increasing_mask;
    b = b.wrapping_sub(&sub_a.horizontal_sum());

    (a, b)
}

/// Fallback function that updates a fletcher checksum.
fn update_fletcher_scalar<BlockType, Iter>(
    mut a: BlockType,
    mut b: BlockType,
    elems: Iter,
) -> (BlockType, BlockType)
where
    BlockType: Copy + Clone + Unsigned + WrappingAdd,
    Iter: Iterator<Item = BlockType>,
{
    for elem in elems {
        a = a.wrapping_add(&elem);
        b = b.wrapping_add(&a);
    }

    (a, b)
}
