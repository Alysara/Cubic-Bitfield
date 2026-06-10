use std::iter::zip;
use std::ops::*;
use std::simd::prelude::*;

use crate::Bitfield;
use crate::simd::NUM_LANES;
use crate::transposes::*;

const SPARSE_THRESHOLD: u32 = 21;
const SPARSE_TO_INDEX_MASK: u32 = !(NUM_LANES as u32 - 1);
const SPARSE_LANE_MASK: u32 = (1 << NUM_LANES) - 1;

#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedBitfield {
    bitfield: Bitfield,
    active_rows: [u32; 32],
    active_slices: u32,
}

// Assign bit ops.
impl BitOrAssign for TrackedBitfield {
    fn bitor_assign(&mut self, rhs: Self) {
        self.process_simd_guided::<false, _>(&rhs, |i, cur| cur | rhs.load_simd(i));
        self.process_rows::<false, _>(|i, cur| cur | rhs.load_rows(i));
        self.active_slices |= rhs.active_slices;
    }
}

impl BitAndAssign for TrackedBitfield {
    fn bitand_assign(&mut self, rhs: Self) {
        self.process_simd::<true, _>(|i, cur| cur & rhs.load_simd(i));
    }
}

impl BitXorAssign for TrackedBitfield {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.process_simd_guided::<true, _>(&rhs, |i, cur| cur ^ rhs.load_simd(i));
    }
}

impl ShlAssign<usize> for TrackedBitfield {
    fn shl_assign(&mut self, rhs: usize) {
        self.process_simd::<true, _>(|_, cur| cur << Simd::splat(rhs as u32));
    }
}

impl ShrAssign<usize> for TrackedBitfield {
    fn shr_assign(&mut self, rhs: usize) {
        self.process_simd::<true, _>(|_, cur| cur >> Simd::splat(rhs as u32));
    }
}

// Derived bit ops (from assign).
impl BitOr for TrackedBitfield {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result |= rhs;
        result
    }
}

impl BitAnd for TrackedBitfield {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result &= rhs;
        result
    }
}

impl BitXor for TrackedBitfield {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = Self::new(0);
        result ^= rhs;
        result
    }
}

impl Shl<usize> for TrackedBitfield {
    type Output = Self;
    fn shl(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        new_bitfield <<= rhs;
        new_bitfield
    }
}

impl Shr<usize> for TrackedBitfield {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut new_bitfield = self;
        new_bitfield >>= rhs;
        new_bitfield
    }
}

impl TrackedBitfield {
    pub fn new(val: u32) -> Self {
        let active_val = if val == 0 { 0 } else { u32::MAX };
        Self {
            bitfield: Bitfield::new(val),
            active_rows: std::array::from_fn(|_| active_val),
            active_slices: active_val,
        }
    }

    pub fn fill(&mut self, val: u32) {
        self.bitfield.fill(val);
        if val == 0 {
            self.active_rows.fill(0);
            self.active_slices = 0;
        } else {
            self.active_rows.fill(u32::MAX);
            self.active_slices = u32::MAX;
        }
    }

    pub fn from_bitfield(bitfield: Bitfield) -> Self {
        let mut tracked = Self {
            bitfield,
            active_rows: [0; 32],
            active_slices: 0,
        };
        tracked.update_tracking();
        tracked
    }

    pub fn as_array(&self) -> &[u32; 1024] {
        self.bitfield.as_array()
    }

    pub fn as_slice(&self) -> &[u32] {
        self.bitfield.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        self.bitfield.as_mut_slice()
    }

    pub fn as_bitfield(&self) -> &Bitfield {
        &self.bitfield
    }

    pub fn as_mut_bitfield(&mut self) -> &mut Bitfield {
        &mut self.bitfield
    }

    #[inline(always)]
    pub fn get_active_slices(&self) -> u32 {
        self.active_slices
    }

    #[inline(always)]
    pub fn get_active_rows(&self, slice: usize) -> u32 {
        self.active_rows[slice]
    }

    #[inline(always)]
    pub fn num_active_slices(&self) -> u32 {
        self.active_slices.count_ones()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_active_slices() == 0
    }

    pub fn active_bit_iter<'a>(&'a self) -> ActiveBitIter<'a> {
        ActiveBitIter::new(self)
    }

    fn simd_active_bit_iter<'a>(&self, rows: &'a [u32; 32]) -> SparseSimdIndexIter<'a> {
        SparseSimdIndexIter {
            active_rows: rows,
            cur_slices: self.active_slices,
            cur_rows: rows[self.active_slices.trailing_zeros() as usize],
            slice: self.active_slices.trailing_zeros(),
        }
    }

    fn active_slice_iter(&self) -> SparseSliceIndexIter {
        SparseSliceIndexIter {
            cur_slices: self.active_slices,
        }
    }

    fn simd_active_slice_iter(&self) -> SparseSimdSliceIndexIter {
        SparseSimdSliceIndexIter {
            cur_slices: self.active_slices,
        }
    }

    #[inline(always)]
    fn process_simd<const TRACK: bool, F: Fn(usize, u32x16) -> u32x16>(&mut self, func: F) {
        if self.is_empty() {
            return;
        };

        if self.num_active_slices() > SPARSE_THRESHOLD {
            self.process_simd_inner::<TRACK, _>(0..32, func);
        } else {
            self.process_simd_inner::<TRACK, _>(self.active_slice_iter(), func);
        }

        if TRACK {
            self.update_all_slices();
        }
    }

    #[inline(always)]
    fn process_simd_guided<const TRACK: bool, F: Fn(usize, u32x16) -> u32x16>(
        &mut self,
        guide: &Self,
        func: F,
    ) {
        if guide.is_empty() {
            return;
        }

        if guide.num_active_slices() > SPARSE_THRESHOLD {
            self.process_simd_inner::<TRACK, _>(0..32, func);
        } else {
            self.process_simd_inner::<TRACK, _>(guide.active_slice_iter(), func);
        }

        if TRACK {
            self.update_all_slices();
        }
    }

    #[inline(always)]
    fn process_simd_inner<const TRACK: bool, F: Fn(usize, u32x16) -> u32x16>(
        &mut self,
        slices: impl Iterator<Item = usize>,
        func: F,
    ) {
        for slice in slices {
            let i = slice * 32;
            let cur1 = self.load_simd(i);
            let cur2 = self.load_simd(i + 16);

            let res1 = func(i, cur1);
            let res2 = func(i + 16, cur2);

            self.store_simd(i, res1);
            self.store_simd(i + 16, res2);

            if TRACK {
                self.update_rows_simd(slice, res1, res2);
            }
        }
    }

    #[inline(always)]
    fn process_rows<const TRACK: bool, F: Fn(usize, u32x16) -> u32x16>(&mut self, func: F) {
        let res1 = func(0, self.load_simd(0));
        let res2 = func(16, self.load_simd(16));

        self.store_rows(0, res1);
        self.store_rows(16, res2);

        if TRACK {
            self.update_slices_simd(res1, res2);
        }
    }

    #[inline(always)]
    fn update_rows_simd(&mut self, slice: usize, res1: u32x16, res2: u32x16) {
        let zero = u32x16::splat(0);
        let bits1 = res1.simd_ne(zero).to_bitmask() as u32;
        let bits2 = res2.simd_ne(zero).to_bitmask() as u32;
        self.active_rows[slice] = bits1 | (bits2 << 16);
    }

    #[inline(always)]
    fn update_slices_simd(&mut self, res1: u32x16, res2: u32x16) {
        let zero = u32x16::splat(0);
        let bits1 = res1.simd_ne(zero).to_bitmask() as u32;
        let bits2 = res2.simd_ne(zero).to_bitmask() as u32;
        self.active_slices = bits1 | (bits2 << 16);
    }

    #[inline(always)]
    fn load_simd(&self, index: usize) -> u32x16 {
        Simd::from_slice(&self.as_slice()[index..])
    }

    #[inline(always)]
    fn load_rows(&self, slice: usize) -> u32x16 {
        Simd::from_slice(&self.active_rows.as_slice()[slice..])
    }

    #[inline(always)]
    fn store_simd(&mut self, index: usize, simd: u32x16) {
        Simd::copy_to_slice(simd, &mut self.as_mut_slice()[index..]);
    }

    #[inline(always)]
    fn store_rows(&mut self, slice: usize, simd: u32x16) {
        Simd::copy_to_slice(simd, &mut self.active_rows.as_mut_slice()[slice..]);
    }

    #[inline(always)]
    fn update_all_slices(&mut self) {
        self.update_slices_simd(self.load_rows(0), self.load_rows(16))
    }

    #[inline(always)]
    fn update_slice_of_rows(&mut self, slice: usize) {
        self.update_rows_simd(
            slice,
            self.load_simd(slice * 32),
            self.load_simd(slice * 32 + 16),
        );
    }

    pub fn update_tracking(&mut self) {
        (0..32).for_each(|i| self.update_slice_of_rows(i));
        self.update_all_slices();
    }

    pub fn andnot(&self, rhs: &Self) -> Self {
        let mut new_bitfield = *self;
        new_bitfield.andnot_assign(rhs);
        new_bitfield
    }

    pub fn andnot_assign(&mut self, rhs: &Self) -> &mut Self {
        self.process_simd_guided::<true, _>(rhs, |i, cur| cur & !rhs.load_simd(i));
        self
    }

    // Core load-into implementations.
    #[inline(always)]
    pub fn load_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        match_val: bool,
    ) {
        self.bitfield
            .load_packed_u1_into::<SET, CMP>(data, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_yz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        x_slice: usize,
        match_val: bool,
    ) {
        self.bitfield
            .load_yz_packed_u1_into::<SET, CMP>(data, x_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_xz_packed_u1_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 512],
        y_slice: usize,
        match_val: bool,
    ) {
        self.bitfield
            .load_xz_packed_u1_into::<SET, CMP>(data, y_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        match_val: u8,
    ) {
        self.bitfield
            .load_packed_u2_into::<SET, CMP>(data, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_yz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        x_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_yz_packed_u2_into::<SET, CMP>(data, x_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_xz_packed_u2_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 1024],
        y_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_xz_packed_u2_into::<SET, CMP>(data, y_slice, match_val);
        self.update_tracking();
    }

    // u4 loaders.
    #[inline(always)]
    pub fn load_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        match_val: u8,
    ) {
        self.bitfield
            .load_packed_u4_into::<SET, CMP>(data, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_yz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        x_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_yz_packed_u4_into::<SET, CMP>(data, x_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_xz_packed_u4_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 2048],
        y_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_xz_packed_u4_into::<SET, CMP>(data, y_slice, match_val);
        self.update_tracking();
    }

    // u8 loaders.
    #[inline(always)]
    pub fn load_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        match_val: u8,
    ) {
        self.bitfield
            .load_packed_u8_into::<SET, CMP>(data, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_yz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        x_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_yz_packed_u8_into::<SET, CMP>(data, x_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_xz_packed_u8_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 4096],
        y_slice: usize,
        match_val: u8,
    ) {
        self.bitfield
            .load_xz_packed_u8_into::<SET, CMP>(data, y_slice, match_val);
        self.update_tracking();
    }

    // u16 loaders.
    #[inline(always)]
    pub fn load_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        match_val: u16,
    ) {
        self.bitfield
            .load_packed_u16_into::<SET, CMP>(data, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_yz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        x_slice: usize,
        match_val: u16,
    ) {
        self.bitfield
            .load_yz_packed_u16_into::<SET, CMP>(data, x_slice, match_val);
        self.update_tracking();
    }

    #[inline(always)]
    pub fn load_xz_packed_u16_into<const SET: u8, const CMP: u8>(
        &mut self,
        data: &[u64; 8192],
        y_slice: usize,
        match_val: u16,
    ) {
        self.bitfield
            .load_xz_packed_u16_into::<SET, CMP>(data, y_slice, match_val);
        self.update_tracking();
    }

    pub fn cull_most_sig_bits(&mut self) -> &mut Self {
        self.process_simd::<true, _>(|_, cur| cur & !(cur << Simd::splat(1)));
        self
    }

    pub fn cull_least_sig_bits(&mut self) -> &mut Self {
        self.process_simd::<true, _>(|_, cur| cur & !(cur >> Simd::splat(1)));
        self
    }

    /// Transposes all 1024 elements as a 32x32 matrix.
    pub fn outer_transpose(&mut self) -> &mut Self {
        self.bitfield.outer_transpose();
        inner_transpose_slice(&mut self.active_rows);
        self.update_all_slices();
        self
    }

    pub fn outer_transpose_scalar(&mut self) -> &mut Self {
        self.bitfield.outer_transpose();
        inner_transpose_slice(&mut self.active_rows);
        self.update_all_slices();
        self
    }

    #[inline(always)]
    pub fn all_ones_slice(&self, index: usize) -> bool {
        let ones = u32x16::splat(u32::MAX);

        let b1 = u32x16::from_slice(&self.bitfield.as_slice()[index..]);
        let b2 = u32x16::from_slice(&self.bitfield.as_slice()[index + 16..]);

        let r1 = b1.simd_eq(ones);
        let r2 = b2.simd_eq(ones);

        (r1 & r2).all()
    }

    #[inline(always)]
    fn is_homogenous_slice(&self, index: usize) -> bool {
        let zero = u32x16::splat(0);
        let ones = u32x16::splat(u32::MAX);

        let b1 = u32x16::from_slice(&self.bitfield.as_slice()[index..]);
        let b2 = u32x16::from_slice(&self.bitfield.as_slice()[index + 16..]);

        let all_zero = b1.simd_eq(zero).all() && b2.simd_eq(zero).all();
        let all_ones = b1.simd_eq(ones).all() && b2.simd_eq(ones).all();

        all_zero || all_ones
    }

    /// Transposes each chunk of 32 elements as a 32x32 bit matrix.
    pub fn inner_transpose(&mut self) -> &mut Self {
        if self.num_active_slices() > SPARSE_THRESHOLD {
            for i in (0..1024).step_by(32) {
                if !self.is_homogenous_slice(i) {
                    inner_transpose_slice(&mut self.bitfield.as_mut_slice()[i..]);
                    self.update_slice_of_rows(i >> 5);
                }
            }
        } else {
            let mut cur_slices = self.active_slices;
            while cur_slices != 0 {
                let slice = cur_slices.trailing_zeros();
                let i = slice as usize * 32;

                if !self.all_ones_slice(i) {
                    inner_transpose_slice(&mut self.bitfield.as_mut_slice()[i..]);
                    self.update_slice_of_rows(slice as usize);
                }

                cur_slices &= cur_slices - 1;
            }
        }
        self
    }

    pub fn inner_transpose_scalar(&mut self) -> &mut Self {
        self.bitfield.inner_transpose_scalar();
        self.update_all_slices();
        self
    }

    pub fn print_inner_slices(&self, range: Range<usize>) {
        self.bitfield.print_inner_slices(range);
    }

    pub fn print_outer_slices(&self, range: Range<usize>) {
        self.bitfield.print_outer_slices(range);
    }
}

struct SparseSimdIndexIter<'a> {
    active_rows: &'a [u32; 32],
    cur_slices: u32,
    cur_rows: u32,
    slice: u32,
}

impl<'a> Iterator for SparseSimdIndexIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.cur_rows != 0 {
                let row = (self.cur_rows.trailing_zeros()) & SPARSE_TO_INDEX_MASK;
                let i = ((self.slice * 32) + row) as usize;
                self.cur_rows &= !(SPARSE_LANE_MASK << row);
                return Some(i);
            }

            if self.cur_slices == 0 {
                return None;
            }

            self.slice = self.cur_slices.trailing_zeros();
            self.cur_rows = self.active_rows[self.slice as usize];
            self.cur_slices &= self.cur_slices - 1;
        }
    }
}

pub struct ActiveBitIter<'a> {
    bitfield: &'a TrackedBitfield,
    cur_slices: u32,
    cur_rows: u32,
    cur_bits: u32,
    slice: u32,
    row: u32,
}

impl<'a> ActiveBitIter<'a> {
    fn new(bitfield: &'a TrackedBitfield) -> Self {
        Self {
            bitfield,
            cur_slices: bitfield.active_slices,
            cur_rows: 0,
            cur_bits: 0,
            slice: 0,
            row: 0,
        }
    }
}

impl<'a> Iterator for ActiveBitIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.cur_bits != 0 {
                let bit = self.cur_bits.trailing_zeros();
                let i = (self.slice * 1024 + self.row * 32 + bit) as usize;
                self.cur_bits &= self.cur_bits - 1;
                return Some(i);
            }

            if self.cur_rows != 0 {
                self.row = self.cur_rows.trailing_zeros();
                let i = self.slice * 32 + self.row;
                self.cur_bits = self.bitfield.as_slice()[i as usize];
                self.cur_rows &= self.cur_rows - 1;
                continue;
            }

            if self.cur_slices != 0 {
                self.slice = self.cur_slices.trailing_zeros();
                self.cur_rows = self.bitfield.active_rows[self.slice as usize];
                self.cur_slices &= self.cur_slices - 1;
                continue;
            }

            return None;
        }
    }
}

struct SparseSimdSliceIndexIter {
    cur_slices: u32,
}

impl Iterator for SparseSimdSliceIndexIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.cur_slices == 0 {
            return None;
        }

        let slice = self.cur_slices.trailing_zeros();
        let i = (slice & SPARSE_TO_INDEX_MASK) as usize;
        self.cur_slices &= !(SPARSE_LANE_MASK << i);
        Some(i)
    }
}

struct SparseSliceIndexIter {
    cur_slices: u32,
}

impl Iterator for SparseSliceIndexIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.cur_slices == 0 {
            return None;
        }

        let slice = self.cur_slices.trailing_zeros();
        self.cur_slices &= self.cur_slices - 1;
        Some(slice as usize)
    }
}
