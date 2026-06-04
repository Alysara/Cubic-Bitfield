use std::mem::transmute;
use std::ops::*;
use std::simd::prelude::*;

#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bitfield {
    data: [u32; 1024],
}

impl BitOr for Bitfield {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        for i in 0..1024 {
            result.data[i] = self.data[i] ^ rhs.data[i];
        }
        result
    }
}

impl BitAnd for Bitfield {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        for i in 0..1024 {
            result.data[i] = self.data[i] & rhs.data[i];
        }
        result
    }
}

impl BitXor for Bitfield {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        for i in 0..1024 {
            result.data[i] = self.data[i] ^ rhs.data[i];
        }
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

impl Bitfield {
    pub fn new(val: u32) -> Self {
        Self {
            data: std::array::from_fn(|_| val),
        }
    }

    pub fn from_packed_u1<const EQ: bool>(array: &[u64; 512]) -> Self {
        let array_u32: &[u32; 1024] = unsafe { transmute(array) };
        if EQ {
            Self {
                data: array_u32.clone(),
            }
        } else {
            let data: [u32; 1024] = std::array::from_fn(|i| !array_u32[i]);
            Self { data }
        }
    }

    #[inline(never)]
    pub fn from_packed_u2<const EQ: bool>(array: &[u64; 1024], match_val: u8) -> Self {
        let mut result = Self::new(0);
        let result_u64: &mut [u64; 512] = unsafe { transmute(&mut result.data) };
        let array_u8: &[u8; 8192] = unsafe { transmute(array) };

        let target1 = u8x64::splat(match_val);
        let target2 = u8x64::splat(match_val << 2);
        let target3 = u8x64::splat(match_val << 4);
        let target4 = u8x64::splat(match_val << 6);

        let mask1 = u8x64::splat(0x03);
        let mask2 = u8x64::splat(0x0C);
        let mask3 = u8x64::splat(0x30);
        let mask4 = u8x64::splat(0xC0);

        let case = if EQ { u8::MAX } else { 0 };
        let true_case = u8x64::splat(case);

        for i in (0..8192).step_by(64) {
            let block = u8x64::from_slice(&array_u8[i..]);
            let b1 = block & mask1;
            let b2 = block & mask2;
            let b3 = block & mask3;
            let b4 = block & mask4;

            // Faster, but needs pdep
            // let pre1 = b1.simd_eq(target1).to_bitmask();
            // let pre2 = b2.simd_eq(target2).to_bitmask();
            // let pre3 = b3.simd_eq(target3).to_bitmask();
            // let pre4 = b4.simd_eq(target4).to_bitmask();
            //
            // let bit_idx = i >> 4; // 256 bits out per 64 bytes = 4 u64s, i/16
            // unsafe {
            //     *result_u64.get_unchecked_mut(bit_idx) =
            //         _pdep_u64(pre1 & 0xFFFF, 0x1111111111111111)
            //             | _pdep_u64(pre2 & 0xFFFF, 0x2222222222222222)
            //             | _pdep_u64(pre3 & 0xFFFF, 0x4444444444444444)
            //             | _pdep_u64(pre4 & 0xFFFF, 0x8888888888888888);
            //     *result_u64.get_unchecked_mut(bit_idx + 1) =
            //         _pdep_u64((pre1 >> 16) & 0xFFFF, 0x1111111111111111)
            //             | _pdep_u64((pre2 >> 16) & 0xFFFF, 0x2222222222222222)
            //             | _pdep_u64((pre3 >> 16) & 0xFFFF, 0x4444444444444444)
            //             | _pdep_u64((pre4 >> 16) & 0xFFFF, 0x8888888888888888);
            //     *result_u64.get_unchecked_mut(bit_idx + 2) =
            //         _pdep_u64((pre1 >> 32) & 0xFFFF, 0x1111111111111111)
            //             | _pdep_u64((pre2 >> 32) & 0xFFFF, 0x2222222222222222)
            //             | _pdep_u64((pre3 >> 32) & 0xFFFF, 0x4444444444444444)
            //             | _pdep_u64((pre4 >> 32) & 0xFFFF, 0x8888888888888888);
            //     *result_u64.get_unchecked_mut(bit_idx + 3) =
            //         _pdep_u64((pre1 >> 48) & 0xFFFF, 0x1111111111111111)
            //             | _pdep_u64((pre2 >> 48) & 0xFFFF, 0x2222222222222222)
            //             | _pdep_u64((pre3 >> 48) & 0xFFFF, 0x4444444444444444)
            //             | _pdep_u64((pre4 >> 48) & 0xFFFF, 0x8888888888888888);
            // }
            let s1_res1 = b1.simd_eq(target1).to_simd();
            let s1_res2 = b2.simd_eq(target2).to_simd();
            let s1_res3 = b3.simd_eq(target3).to_simd();
            let s1_res4 = b4.simd_eq(target4).to_simd();

            let (s2_res1, s2_res2): (u16x32, u16x32) =
                unsafe { transmute(s1_res1.interleave(s1_res2)) };
            let (s2_res3, s2_res4): (u16x32, u16x32) =
                unsafe { transmute(s1_res3.interleave(s1_res4)) };

            let (s3_res1, s3_res2): (u8x64, u8x64) =
                unsafe { transmute(s2_res1.interleave(s2_res3)) };
            let (s3_res3, s3_res4): (u8x64, u8x64) =
                unsafe { transmute(s2_res2.interleave(s2_res4)) };

            let bits1 = s3_res1.simd_eq(true_case);
            let bits2 = s3_res2.simd_eq(true_case);
            let bits3 = s3_res3.simd_eq(true_case);
            let bits4 = s3_res4.simd_eq(true_case);

            let bit_idx = i >> 4;
            unsafe {
                *result_u64.get_unchecked_mut(bit_idx) = bits1.to_bitmask();
                *result_u64.get_unchecked_mut(bit_idx + 1) = bits2.to_bitmask();
                *result_u64.get_unchecked_mut(bit_idx + 2) = bits3.to_bitmask();
                *result_u64.get_unchecked_mut(bit_idx + 3) = bits4.to_bitmask();
            }
        }
        result
    }

    #[inline(never)]
    pub fn from_packed_u4<const EQ: bool>(array: &[u64; 2048], match_val: u8) -> Self {
        let mut result = Self::new(0);
        let result_u64: &mut [u64; 512] = unsafe { transmute(&mut result.data) };
        let array_u8: &[u8; 16384] = unsafe { transmute(array) };

        let target1 = u8x64::splat(match_val);
        let target2 = u8x64::splat(match_val << 4);

        let mask1 = u8x64::splat(0x0F);
        let mask2 = u8x64::splat(0xF0);

        for i in (0..16384).step_by(64) {
            let block = u8x64::from_slice(&array_u8[i..]);
            let a1 = block & mask1;
            let a2 = block & mask2;

            let b1 = a1.simd_eq(target1).to_simd();
            let b2 = a2.simd_eq(target2).to_simd();

            let (c1, c2) = b1.interleave(b2);

            let case: i8 = if EQ { -1 } else { 0 };
            let true_case = i8x64::splat(case);

            let bits1 = c1.simd_eq(true_case);
            let bits2 = c2.simd_eq(true_case);
            // println!("c1: {:?}", c1);
            // println!("c2: {:?}", c2);

            let bit_idx = i >> 5;
            // println!("index: {bit_idx}");
            unsafe {
                *result_u64.get_unchecked_mut(bit_idx) = bits1.to_bitmask();
                *result_u64.get_unchecked_mut(bit_idx + 1) = bits2.to_bitmask();
            }
        }
        result
    }

    #[inline(never)]
    pub fn from_packed_u8<const EQ: bool>(array: &[u64; 4096], match_val: u8) -> Self {
        let mut result = Self::new(0);
        let result_u64: &mut [u64; 512] = unsafe { transmute(&mut result.data) };
        let array_u8: &[u8; 32768] = unsafe { transmute(array) };

        let target = u8x64::splat(match_val);
        for i in (0..32768).step_by(64) {
            let block = u8x64::from_slice(&array_u8[i..]);
            let bits = if EQ {
                block.simd_eq(target).to_bitmask()
            } else {
                block.simd_ne(target).to_bitmask()
            };

            let bit_idx = i >> 6;
            unsafe { *result_u64.get_unchecked_mut(bit_idx) = bits };
        }
        result
    }

    #[inline(never)]
    pub fn from_packed_u16<const EQ: bool>(array: &[u64; 8192], match_val: u16) -> Self {
        let mut result = Self::new(0);
        let array_u16: &[u16; 32768] = unsafe { transmute(array) };

        let target = u16x32::splat(match_val);
        for i in (0..32768).step_by(32) {
            let block = u16x32::from_slice(&array_u16[i..]);
            let bits = if EQ {
                block.simd_eq(target).to_bitmask()
            } else {
                block.simd_ne(target).to_bitmask()
            };

            let bit_idx = i >> 5;
            unsafe { *result.data.get_unchecked_mut(bit_idx) = bits as u32 };
        }
        result
    }

    pub fn as_array(self) -> [u32; 1024] {
        self.data
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.data
    }

    pub fn cull_most_sig_bits(&mut self) {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block << u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
    }

    pub fn cull_least_sig_bits(&mut self) {
        for i in (0..1024).step_by(16) {
            let block = u32x16::from_slice(&self.data[i..i + 16]);
            let culled = block & !(block >> u32x16::splat(1));
            culled.copy_to_slice(&mut self.data[i..i + 16]);
        }
    }

    /// Transposes all 1024 elements as a 32x32 matrix.
    pub fn outer_transpose(&mut self) {
        for y in 0..8 {
            // Handle middle block case.
            let i = y * 128 + y * 4;
            let block = self.load_4x4_block(i);
            let tblock = Self::transpose_4x4_block(block);
            self.store_4x4_block(i, tblock);

            for z in (y + 1)..8 {
                // Perform 4x4 block swaps.
                let i = y * 128 + z * 4;
                let j = y * 4 + z * 128;

                let i_block = self.load_4x4_block(i);
                let j_block = self.load_4x4_block(j);

                let i_tblock = Self::transpose_4x4_block(i_block);
                let j_tblock = Self::transpose_4x4_block(j_block);

                self.store_4x4_block(i, j_tblock);
                self.store_4x4_block(j, i_tblock);
            }
        }
    }

    #[inline(always)]
    fn load_4x4_block(&self, index: usize) -> (u32x4, u32x4, u32x4, u32x4) {
        (
            u32x4::from_slice(&self.data[index..]),
            u32x4::from_slice(&self.data[index + 32..]),
            u32x4::from_slice(&self.data[index + 64..]),
            u32x4::from_slice(&self.data[index + 96..]),
        )
    }

    #[inline(always)]
    fn store_4x4_block(&mut self, index: usize, block: (u32x4, u32x4, u32x4, u32x4)) {
        block.0.copy_to_slice(&mut self.data[index..]);
        block.1.copy_to_slice(&mut self.data[index + 32..]);
        block.2.copy_to_slice(&mut self.data[index + 64..]);
        block.3.copy_to_slice(&mut self.data[index + 96..]);
    }

    fn transpose_4x4_block(block: (u32x4, u32x4, u32x4, u32x4)) -> (u32x4, u32x4, u32x4, u32x4) {
        const SWIZZLE_64_LO: [usize; 4] = [0, 1, 4, 5];
        const SWIZZLE_64_HI: [usize; 4] = [2, 3, 6, 7];
        const SWIZZLE_32_LO: [usize; 4] = [0, 4, 2, 6];
        const SWIZZLE_32_HI: [usize; 4] = [1, 5, 3, 7];

        let t1 = simd_swizzle!(block.0, block.2, SWIZZLE_64_LO);
        let t2 = simd_swizzle!(block.1, block.3, SWIZZLE_64_LO);
        let t3 = simd_swizzle!(block.0, block.2, SWIZZLE_64_HI);
        let t4 = simd_swizzle!(block.1, block.3, SWIZZLE_64_HI);

        (
            simd_swizzle!(t1, t2, SWIZZLE_32_LO),
            simd_swizzle!(t1, t2, SWIZZLE_32_HI),
            simd_swizzle!(t3, t4, SWIZZLE_32_LO),
            simd_swizzle!(t3, t4, SWIZZLE_32_HI),
        )
    }

    #[inline(never)]
    pub fn outer_transpose_scalar(&mut self) {
        for z in 0..32 {
            for y in (z + 1)..32 {
                let tmp = self.data[z * 32 + y];
                self.data[z * 32 + y] = self.data[y * 32 + z];
                self.data[y * 32 + z] = tmp;
            }
        }
    }

    // --- Inner transpose algorithm obtained from:
    // --- https://github.com/Pnoenix/fast-bit-matrix-transpose

    /// Transposes each chunk of 32 elements as a 32x32 bit matrix.
    pub fn inner_transpose(&mut self) {
        for i in (0..1024).step_by(32) {
            // 16 bits
            let hi = u32x16::from_slice(&self.data[i..]);
            let lo = u32x16::from_slice(&self.data[i + 16..]);
            const MASK_16: u32x16 = u32x16::splat(0x0000FFFF);
            const SHIFT_16: u32x16 = u32x16::splat(16);
            let hi16 = (hi & MASK_16) | lo << SHIFT_16;
            let lo16 = (lo & !MASK_16) | hi >> SHIFT_16;

            // 8 bits
            const S8H: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
            const S8L: [usize; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
            let hi8_prep = simd_swizzle!(hi16, lo16, S8H);
            let lo8_prep = simd_swizzle!(hi16, lo16, S8L);
            const MASK_8: u32x16 = u32x16::splat(0x00FF00FF);
            const SHIFT_8: u32x16 = u32x16::splat(8);
            let hi8 = (hi8_prep & MASK_8) | ((lo8_prep & MASK_8) << SHIFT_8);
            let lo8 = (lo8_prep & !MASK_8) | ((hi8_prep & !MASK_8) >> SHIFT_8);

            // 4 bits
            const S4H: [usize; 16] = [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27];
            const S4L: [usize; 16] = [4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31];
            let hi4_prep = simd_swizzle!(hi8, lo8, S4H);
            let lo4_prep = simd_swizzle!(hi8, lo8, S4L);
            const MASK_4: u32x16 = u32x16::splat(0x0F0F0F0F);
            const SHIFT_4: u32x16 = u32x16::splat(4);
            let hi4 = (hi4_prep & MASK_4) | ((lo4_prep & MASK_4) << SHIFT_4);
            let lo4 = (lo4_prep & !MASK_4) | ((hi4_prep & !MASK_4) >> SHIFT_4);

            // 2 bits
            const S2H: [usize; 16] = [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29];
            const S2L: [usize; 16] = [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31];
            let hi2_prep = simd_swizzle!(hi4, lo4, S2H);
            let lo2_prep = simd_swizzle!(hi4, lo4, S2L);
            const MASK_2: u32x16 = u32x16::splat(0x33333333);
            const SHIFT_2: u32x16 = u32x16::splat(2);
            let hi2 = (hi2_prep & MASK_2) | ((lo2_prep & MASK_2) << SHIFT_2);
            let lo2 = (lo2_prep & !MASK_2) | ((hi2_prep & !MASK_2) >> SHIFT_2);

            // 1 bit
            const S1H: [usize; 16] = [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
            const S1L: [usize; 16] = [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31];
            let hi1_prep = simd_swizzle!(hi2, lo2, S1H);
            let lo1_prep = simd_swizzle!(hi2, lo2, S1L);
            const MASK_1: u32x16 = u32x16::splat(0x55555555);
            const SHIFT_1: u32x16 = u32x16::splat(1);
            let hi1 = (hi1_prep & MASK_1) | ((lo1_prep & MASK_1) << SHIFT_1);
            let lo1 = (lo1_prep & !MASK_1) | ((hi1_prep & !MASK_1) >> SHIFT_1);

            // Final swizzle
            const SFH: [usize; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
            const SFL: [usize; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
            let final_hi = simd_swizzle!(hi1, lo1, SFH);
            let final_lo = simd_swizzle!(hi1, lo1, SFL);
            final_hi.copy_to_slice(&mut self.data[i..]);
            final_lo.copy_to_slice(&mut self.data[i + 16..]);
        }
    }

    pub fn inner_transpose_scalar(&mut self) {
        for i in 0..32 {
            for j in 0..32 {
                for k in (j + 1)..32 {
                    let swap = ((self.data[i * 32 + j] >> k) ^ (self.data[i * 32 + k] >> j)) & 1;
                    self.data[i * 32 + j] ^= swap << k;
                    self.data[i * 32 + k] ^= swap << j;
                }
            }
        }
    }

    pub fn print_inner_slices(&self, range: Range<usize>) {
        assert!(
            range.end <= 32,
            "End of range is too large! {} is not <= 32.",
            range.end
        );
        assert!(
            range.start < 32,
            "Start of range is too large! {} is not < 32.",
            range.start
        );

        for slice in range {
            println!("{:-<35}", format!("|- Inner slice {slice} "));

            for i in 0..32 {
                let index = slice * 32 + i;
                let prefix = if i % 2 == 0 { "+" } else { "|" };
                println!("{prefix} {:032b}", self.data[index]);
            }
            println!("{:-<35}", "|");
            println!();
        }
    }

    pub fn print_outer_slices(&self, range: Range<usize>) {
        assert!(
            range.end <= 32,
            "End of range is too large! {} is not <= 32.",
            range.end
        );
        assert!(
            range.start < 32,
            "Start of range is too large! {} is not < 32.",
            range.start
        );

        for slice in range {
            println!("{:-<35}", format!("|- Outer slice {slice} "));

            for i in 0..32 {
                let mut bits = 0;
                for j in 0..32 {
                    let index = i * 32 + j;
                    let bit = (self.data[index] >> slice) & 1;
                    bits |= bit << j
                }
                let prefix = if i % 2 == 0 { "+" } else { "|" };
                println!("{prefix} {:032b}", bits);
            }
            println!("{:-<35}", "|");
            println!();
        }
    }
}
