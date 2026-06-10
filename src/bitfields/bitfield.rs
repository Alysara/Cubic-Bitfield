use std::mem::transmute;
use std::ops::*;
use std::simd::prelude::*;

use crate::loader::*;
use crate::logger::*;
use crate::transposes::*;

#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitfield {
    data: [u32; 1024],
}

// Bit operations.

impl BitOr for Bitfield {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result |= rhs;
        result
    }
}

impl BitAnd for Bitfield {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result &= rhs;
        result
    }
}

impl BitXor for Bitfield {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result ^= rhs;
        result
    }
}

impl BitOrAssign for Bitfield {
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] |= rhs.data[i];
        }
    }
}

impl BitAndAssign for Bitfield {
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] &= rhs.data[i];
        }
    }
}

impl BitXorAssign for Bitfield {
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..1024 {
            self.data[i] ^= rhs.data[i];
        }
    }
}

impl Shl<usize> for Bitfield {
    type Output = Self;
    fn shl(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        for i in 0..1024 {
            new_bitfield.data[i] <<= rhs;
        }
        new_bitfield
    }
}

impl Shr<usize> for Bitfield {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        for i in 0..1024 {
            new_bitfield.data[i] >>= rhs;
        }
        new_bitfield
    }
}

impl ShlAssign<usize> for Bitfield {
    fn shl_assign(&mut self, rhs: usize) {
        for i in 0..1024 {
            self.data[i] <<= rhs;
        }
    }
}

impl ShrAssign<usize> for Bitfield {
    fn shr_assign(&mut self, rhs: usize) {
        for i in 0..1024 {
            self.data[i] >>= rhs;
        }
    }
}

impl Bitfield {
    pub fn new(val: u32) -> Self {
        Self {
            data: std::array::from_fn(|_| val),
        }
    }

    pub fn fill(&mut self, val: u32) {
        for i in 0..1024 {
            self.data[i] = val;
        }
    }

    fn set_u32<const SET: u8>(&mut self, index: usize, value: u32) {
        apply_set::<SET, u32>(&mut self.data[index], value);
    }

    fn set_u64<const SET: u8>(&mut self, index: usize, value: u64) {
        let bitfield_64: &mut [u64; 1024] = unsafe { transmute(&mut self.data) };
        apply_set::<SET, u64>(&mut bitfield_64[index], value);
    }

    pub fn andnot(&self, rhs: &Self) -> Self {
        let mut new_bitfield = *self;
        for i in 0..1024 {
            new_bitfield.data[i] &= !rhs.data[i];
        }
        new_bitfield
    }

    pub fn andnot_assign(&mut self, rhs: &Self) -> &mut Self {
        for i in 0..1024 {
            self.data[i] &= !rhs.data[i];
        }
        self
    }

    // Core load-into implementations.

    // u1 loaders.
    #[inline(always)]
    fn load_packed_u1_inner<'a, const SET: u8, const CMP: u8, const MATCH: bool>(
        &mut self,
        src: impl Iterator<Item = (usize, &'a u32)>,
    ) {
        for (i, val) in src {
            self.set_u32::<SET>(i, process_packed_u1::<CMP, MATCH>(val));
        }
    }

    #[inline(always)]
    pub fn load_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate();
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    #[inline(always)]
    pub fn load_yz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        x_slice: usize,
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate().skip(x_slice * 32).take(32);
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    #[inline(always)]
    pub fn load_xz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        y_slice: usize,
        match_val: bool,
    ) {
        let data_32: &[u32; 1024] = unsafe { transmute(data) };
        let iter = data_32.iter().enumerate().skip(y_slice).step_by(32);
        match match_val {
            true => self.load_packed_u1_inner::<SET, CMP, true>(iter),
            false => self.load_packed_u1_inner::<SET, CMP, false>(iter),
        }
    }

    // u2 loaders.
    #[inline(always)]
    fn load_packed_u2_inner<const SET: u8, const CMP: u8, const MATCH: u8>(
        &mut self,
        data: &[u64; 1024],
        src: impl Iterator<Item = usize>,
    ) {
        for i in src {
            let (a, b, c, d) = process_packed_u2::<CMP, MATCH>(&data[i..]);

            let b_idx = i >> 1;
            self.set_u64::<SET>(b_idx, a);
            self.set_u64::<SET>(b_idx + 1, b);
            self.set_u64::<SET>(b_idx + 2, c);
            self.set_u64::<SET>(b_idx + 3, d);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u2_inner<const SET: u8, const CMP: u8, const MATCH: u8>(
        &mut self,
        data: &[u64; 1024],
        src: impl Iterator<Item = usize>,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i],       data[i + 32],  data[i + 64],  data[i + 96],
                data[i + 128], data[i + 160], data[i + 192], data[i + 224],
            ];
            let (a, b, c, d) = process_packed_u2::<CMP, MATCH>(&slice);

            self.set_u32::<SET>(i, a as u32);
            self.set_u32::<SET>(i + 32, (a >> 32) as u32);
            self.set_u32::<SET>(i + 64, b as u32);
            self.set_u32::<SET>(i + 96, (b >> 32) as u32);
            self.set_u32::<SET>(i + 128, c as u32);
            self.set_u32::<SET>(i + 160, (c >> 32) as u32);
            self.set_u32::<SET>(i + 192, d as u32);
            self.set_u32::<SET>(i + 224, (d >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        match_val: u8,
    ) {
        let iter = (0..1024).step_by(8);
        match match_val {
            0 => self.load_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    #[inline(always)]
    pub fn load_yz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..1024).skip(x_slice * 32).step_by(8).take(4);
        match match_val {
            0 => self.load_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    #[inline(always)]
    pub fn load_xz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..1024).skip(y_slice).step_by(256);
        match match_val {
            0 => self.load_xz_packed_u2_inner::<SET, CMP, 0>(data, iter),
            1 => self.load_xz_packed_u2_inner::<SET, CMP, 1>(data, iter),
            2 => self.load_xz_packed_u2_inner::<SET, CMP, 2>(data, iter),
            3 => self.load_xz_packed_u2_inner::<SET, CMP, 3>(data, iter),
            _ => unreachable!(),
        };
    }

    // u4 loaders.
    #[inline(always)]
    fn load_packed_u4_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            let (a, b) = process_packed_u4::<CMP>(&data[i..], cmp_val);

            let b_idx = i >> 2;
            self.set_u64::<SET>(b_idx, a);
            self.set_u64::<SET>(b_idx + 1, b);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u4_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i], data[i + 1], data[i + 64], data[i + 65],
                data[i + 128], data[i + 129], data[i + 192], data[i + 193]
            ];
            let (a, b) = process_packed_u4::<CMP>(&slice, cmp_val);

            let bitfield_idx = i >> 1;
            self.set_u32::<SET>(bitfield_idx, a as u32);
            self.set_u32::<SET>(bitfield_idx + 32, (a >> 32) as u32);
            self.set_u32::<SET>(bitfield_idx + 64, b as u32);
            self.set_u32::<SET>(bitfield_idx + 96, (b >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        match_val: u8,
    ) {
        let iter = (0..2048).step_by(8);
        self.load_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..2048).skip(x_slice * 64).step_by(8).take(8);
        self.load_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..2048).skip(y_slice * 2).step_by(256);
        self.load_xz_packed_u4_inner::<SET, CMP>(data, iter, match_val);
    }

    // u8 loaders.
    #[inline(always)]
    fn load_packed_u8_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            let bits = process_packed_u8::<CMP>(&data[i..], cmp_val);
            self.set_u64::<SET>(i >> 3, bits);
        }
    }

    #[inline(always)]
    fn load_xz_packed_u8_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        src: impl Iterator<Item = usize>,
        cmp_val: u8,
    ) {
        for i in src {
            #[rustfmt::skip]
            let slice = [
                data[i], data[i + 1], data[i + 2], data[i + 3],
                data[i + 128], data[i + 129], data[i + 130], data[i + 131],
            ];
            let bits = process_packed_u8::<CMP>(&slice, cmp_val);

            let b_idx = i >> 2;
            self.set_u32::<SET>(b_idx, bits as u32);
            self.set_u32::<SET>(b_idx + 32, (bits >> 32) as u32);
        }
    }

    #[inline(always)]
    pub fn load_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        match_val: u8,
    ) {
        let iter = (0..4096).step_by(8);
        self.load_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        x_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..4096).skip(x_slice * 128).step_by(8).take(16);
        self.load_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        y_slice: usize,
        match_val: u8,
    ) {
        let iter = (0..4096).skip(y_slice * 4).step_by(256);
        self.load_xz_packed_u8_inner::<SET, CMP>(data, iter, match_val);
    }

    // u16 loaders.
    #[inline(always)]
    fn load_packed_u16_inner<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        src: impl Iterator<Item = usize>,
        cmp_val: u16,
    ) {
        for i in src {
            let bits = process_packed_u16::<CMP>(&data[i..], cmp_val);
            self.set_u32::<SET>(i >> 3, bits);
        }
    }

    #[inline(always)]
    pub fn load_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        match_val: u16,
    ) {
        let iter = (0..8192).step_by(8);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_yz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        x_slice: usize,
        match_val: u16,
    ) {
        let iter = (0..8192).skip(x_slice * 256).step_by(8).take(32);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    #[inline(always)]
    pub fn load_xz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        y_slice: usize,
        match_val: u16,
    ) {
        let iter = (0..8192).skip(y_slice * 8).step_by(256);
        self.load_packed_u16_inner::<SET, CMP>(data, iter, match_val);
    }

    pub fn to_array(self) -> [u32; 1024] {
        self.data
    }

    pub fn to_array_64(self) -> [u64; 512] {
        unsafe { transmute(self.data) }
    }

    pub fn as_array(&self) -> &[u32; 1024] {
        &self.data
    }

    pub fn as_slice(&self) -> &[u32] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        self.data.as_mut_slice()
    }

    pub fn cull_most_sig_bits(&mut self) -> &mut Self {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block << u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
        self
    }

    pub fn cull_least_sig_bits(&mut self) -> &mut Self {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block >> u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
        self
    }

    /// Transposes all 1024 elements as a 32x32 matrix.
    pub fn outer_transpose(&mut self) -> &mut Self {
        outer_transpose(&mut self.data);
        self
    }

    pub fn outer_transpose_scalar(&mut self) -> &mut Self {
        outer_transpose_scalar(&mut self.data);
        self
    }

    /// Transposes each chunk of 32 elements as a 32x32 bit matrix.
    pub fn inner_transpose(&mut self) -> &mut Self {
        for i in (0..1024).step_by(32) {
            inner_transpose_slice(&mut self.data[i..]);
        }
        self
    }

    pub fn inner_transpose_scalar(&mut self) -> &mut Self {
        for i in (0..1024).step_by(32) {
            inner_transpose_slice_scalar(&mut self.data[i..]);
        }
        self
    }

    pub fn print_inner_slices(&self, range: Range<usize>) {
        print_matrix_inner_slices(&self.data, range);
    }

    pub fn print_outer_slices(&self, range: Range<usize>) {
        print_matrix_outer_slices(&self.data, range);
    }
}
