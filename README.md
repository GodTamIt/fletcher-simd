# fletcher-simd

[![crates.io](https://img.shields.io/crates/v/fletcher-simd.svg)](https://crates.io/crates/fletcher-simd)
[![crates.io](https://img.shields.io/badge/license-BSD%202--Clause-blue)](https://crates.io/crates/fletcher-simd)
[![docs.rs](https://img.shields.io/docsrs/fletcher-simd)](https://docs.rs/fletcher-simd/)

A SIMD implementation of the [Fletcher's checksum] algorithm.

**Note:** This implementation uses a modulus of `2^k` where `k` is the checksum block size in bits, as this is fast with wrapping math. Other implementations may use `2^k - 1`.

## Features

  * Uses `std::simd`, which currently requires **nightly**.
  * Supports all architectures supported by `std::simd`.
  * Both run-time and compile-time detection available via the [`multiversion`] crate.
  * Scalar fallback.

## Example

```rust
use byteorder::{ByteOrder, LittleEndian};
use fletcher_simd::Fletcher128;

fn main() {
    const DATA: &str = "abcdefgh";
    let mut fletcher = Fletcher128::new();

    // Read bytes in little endian. Endianness matters!
    fletcher.update_with_iter(
        DATA.as_bytes()
            .chunks(8)
            .map(|chunk| LittleEndian::read_u64(chunk)),
    );

    assert_eq!(fletcher.value(), 0x68676665646362616867666564636261);
}
```

[Fletcher's checksum]: https://en.wikipedia.org/wiki/Fletcher's_checksum
[`multiversion`]: https://crates.io/crates/multiversion