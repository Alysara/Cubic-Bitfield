#![feature(portable_simd)]

pub(crate) mod bitfields {
    pub mod bitfield;
    pub mod tracked_bitfield;
}

pub(crate) mod transposes;
pub(crate) mod loader;
pub(crate) mod logger;
pub(crate) mod simd;
pub mod util;

pub use bitfields::bitfield::Bitfield;
pub use bitfields::tracked_bitfield::TrackedBitfield;
pub use loader::*;
