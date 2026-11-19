//! Systematic quadratic quotient ring codes over the Wiedemann binary tower.
#![no_std]

extern crate alloc;

mod encoder;
mod kernel;
mod naive;
mod omega;
mod product;

pub use encoder::{QaEncoder, interleaved_encode_batch, subfield_encode_batch};
pub use kernel::{forward_transform, inverse_transform};
pub use naive::{NaiveQaEncoder, naive_forward, naive_inverse, quotient_product};
pub use omega::{QaButterflyValue, QaField};

/// Validate the code geometry before using it to split buffers.
fn geometry(log_message_len: usize, log_inv_rate: usize) -> (usize, usize) {
    assert!(log_message_len > 0 && log_message_len < usize::BITS as usize);
    assert!(log_inv_rate < usize::BITS as usize - log_message_len);
    (1 << log_message_len, (1 << log_inv_rate) - 1)
}
