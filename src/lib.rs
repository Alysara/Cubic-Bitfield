#![feature(portable_simd)]

pub(crate) mod bitfields {
    pub mod bitfield;
}

pub(crate) mod transposes;
pub(crate) mod loader;
pub mod util;

pub use bitfields::bitfield::Bitfield;
pub use loader::*;
