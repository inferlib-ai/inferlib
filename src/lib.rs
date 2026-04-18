//! inferlib-core: Fast LLM inference primitives
//!
//! Contains: sampling kernels (top-k, nucleus/top-p, min-p, typical, greedy)
//!
//! # Example
//! ```
//! use inferlib_core::sampling::{sample, apply_temperature};
//!
//! let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let token = sample(
//!     &mut logits,
//!     "nucleus",
//!     0.7,
//!     Some(0.9),
//!     Some(50),
//!     None,
//!     None,
//!     None,
//!     Some(42),
//! );
//! ```

pub mod sampling;

pub use sampling::sample;
