#![feature(portable_simd)]

pub(crate) mod bitfields {
    pub mod bitfield;
}

pub use bitfields::bitfield::Bitfield;
