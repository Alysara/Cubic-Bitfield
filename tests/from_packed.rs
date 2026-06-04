use std::mem::transmute;

use cubic_bitfield::Bitfield;

#[test]
fn from_packed_u1_test() {
    let mut array: [u64; 512] = std::array::from_fn(|_| 0);
    for i in 0..512 {
        array[i] = (i as u64).wrapping_mul(21312312313242);
    }
    let bitfield = Bitfield::from_packed_u1::<true>(&array);
    let slice_u32: [u32; 1024] = bitfield.as_array();
    let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
    let array_u8: &[u8; 4096] = unsafe { transmute(&array) };
    for i in 0..4096 {
        assert_eq!(slice[i], array_u8[i], "u1 true i={i}");
    }
}

#[test]
fn from_packed_u1_false_test() {
    let mut array: [u64; 512] = std::array::from_fn(|_| 0);
    for i in 0..512 {
        array[i] = (i as u64).wrapping_mul(21312312313242);
    }
    let bitfield = Bitfield::from_packed_u1::<false>(&array);
    let slice_u32: [u32; 1024] = bitfield.as_array();
    let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
    let array_u8: &[u8; 4096] = unsafe { transmute(&array) };
    for i in 0..4096 {
        assert_eq!(slice[i], !array_u8[i], "u1 false i={i}");
    }
}

#[test]
fn from_packed_u2_test() {
    let array_u8: [u8; 8192] = std::array::from_fn(|i| {
        let v = (i % 4) as u8;
        v | (v.wrapping_add(1) % 4 << 2)
            | (v.wrapping_add(2) % 4 << 4)
            | (v.wrapping_add(3) % 4 << 6)
    });
    let array: [u64; 1024] = unsafe { transmute(array_u8) };

    for target in 0u8..4 {
        let bitfield = Bitfield::from_packed_u2::<true>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..8192usize {
            let byte = array_u8[i];
            let b0 = (byte & 0x03) == target;
            let b1 = ((byte >> 2) & 0x03) == target;
            let b2 = ((byte >> 4) & 0x03) == target;
            let b3 = ((byte >> 6) & 0x03) == target;
            let bit_idx = i * 4;
            let out_byte = slice[bit_idx / 8];
            let bit_off = bit_idx % 8;
            assert_eq!(
                (out_byte >> bit_off) & 1 == 1,
                b0,
                "u2 true i={i} target={target} b0"
            );
            assert_eq!(
                (out_byte >> (bit_off + 1)) & 1 == 1,
                b1,
                "u2 true i={i} target={target} b1"
            );
            assert_eq!(
                (out_byte >> (bit_off + 2)) & 1 == 1,
                b2,
                "u2 true i={i} target={target} b2"
            );
            assert_eq!(
                (out_byte >> (bit_off + 3)) & 1 == 1,
                b3,
                "u2 true i={i} target={target} b3"
            );
        }
    }
}

#[test]
fn from_packed_u2_false_test() {
    let array_u8: [u8; 8192] = std::array::from_fn(|i| {
        let v = (i % 4) as u8;
        v | (v.wrapping_add(1) % 4 << 2)
            | (v.wrapping_add(2) % 4 << 4)
            | (v.wrapping_add(3) % 4 << 6)
    });
    let array: [u64; 1024] = unsafe { transmute(array_u8) };

    for target in 0u8..4 {
        let bitfield = Bitfield::from_packed_u2::<false>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..8192usize {
            let byte = array_u8[i];
            let b0 = (byte & 0x03) != target;
            let b1 = ((byte >> 2) & 0x03) != target;
            let b2 = ((byte >> 4) & 0x03) != target;
            let b3 = ((byte >> 6) & 0x03) != target;
            let bit_idx = i * 4;
            let out_byte = slice[bit_idx / 8];
            let bit_off = bit_idx % 8;
            assert_eq!(
                (out_byte >> bit_off) & 1 == 1,
                b0,
                "u2 false i={i} target={target} b0"
            );
            assert_eq!(
                (out_byte >> (bit_off + 1)) & 1 == 1,
                b1,
                "u2 false i={i} target={target} b1"
            );
            assert_eq!(
                (out_byte >> (bit_off + 2)) & 1 == 1,
                b2,
                "u2 false i={i} target={target} b2"
            );
            assert_eq!(
                (out_byte >> (bit_off + 3)) & 1 == 1,
                b3,
                "u2 false i={i} target={target} b3"
            );
        }
    }
}

#[test]
fn from_packed_u4_test() {
    let array_u8: [u8; 16384] = std::array::from_fn(|i| {
        let lo = (i % 16) as u8;
        let hi = ((i + 7) % 16) as u8;
        lo | (hi << 4)
    });
    let array: [u64; 2048] = unsafe { transmute(array_u8) };

    for target in [0u8, 3, 7, 11, 15] {
        let bitfield = Bitfield::from_packed_u4::<true>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..16384usize {
            let byte = array_u8[i];
            let b0 = (byte & 0x0F) == target;
            let b1 = ((byte >> 4) & 0x0F) == target;
            let bit_idx = i * 2;
            let out_byte = slice[bit_idx / 8];
            let bit_off = bit_idx % 8;
            assert_eq!(
                (out_byte >> bit_off) & 1 == 1,
                b0,
                "u4 true i={i} target={target} b0"
            );
            assert_eq!(
                (out_byte >> (bit_off + 1)) & 1 == 1,
                b1,
                "u4 true i={i} target={target} b1"
            );
        }
    }
}

#[test]
fn from_packed_u4_false_test() {
    let array_u8: [u8; 16384] = std::array::from_fn(|i| {
        let lo = (i % 16) as u8;
        let hi = ((i + 7) % 16) as u8;
        lo | (hi << 4)
    });
    let array: [u64; 2048] = unsafe { transmute(array_u8) };

    for target in [0u8, 3, 7, 11, 15] {
        let bitfield = Bitfield::from_packed_u4::<false>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..16384usize {
            let byte = array_u8[i];
            let b0 = (byte & 0x0F) != target;
            let b1 = ((byte >> 4) & 0x0F) != target;
            let bit_idx = i * 2;
            let out_byte = slice[bit_idx / 8];
            let bit_off = bit_idx % 8;
            assert_eq!(
                (out_byte >> bit_off) & 1 == 1,
                b0,
                "u4 false i={i} target={target} b0"
            );
            assert_eq!(
                (out_byte >> (bit_off + 1)) & 1 == 1,
                b1,
                "u4 false i={i} target={target} b1"
            );
        }
    }
}

#[test]
fn from_packed_u8_test() {
    let array_u8: [u8; 32768] = std::array::from_fn(|i| {
        ((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            % 256) as u8
    });
    let array: [u64; 4096] = unsafe { transmute(array_u8) };

    for target in [0u8, 42, 127, 200, 255] {
        let bitfield = Bitfield::from_packed_u8::<true>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..32768usize {
            let expected = array_u8[i] == target;
            let out_byte = slice[i / 8];
            let actual = (out_byte >> (i % 8)) & 1 == 1;
            assert_eq!(actual, expected, "u8 true i={i} target={target}");
        }
    }
}

#[test]
fn from_packed_u8_false_test() {
    let array_u8: [u8; 32768] = std::array::from_fn(|i| {
        ((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            % 256) as u8
    });
    let array: [u64; 4096] = unsafe { transmute(array_u8) };

    for target in [0u8, 42, 127, 200, 255] {
        let bitfield = Bitfield::from_packed_u8::<false>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..32768usize {
            let expected = array_u8[i] != target;
            let out_byte = slice[i / 8];
            let actual = (out_byte >> (i % 8)) & 1 == 1;
            assert_eq!(actual, expected, "u8 false i={i} target={target}");
        }
    }
}

#[test]
fn from_packed_u16_test() {
    let array_u16: [u16; 32768] = std::array::from_fn(|i| {
        ((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            % 512) as u16
    });
    let array: [u64; 8192] = unsafe { transmute(array_u16) };

    for target in [0u16, 7, 63, 255, 511] {
        let bitfield = Bitfield::from_packed_u16::<true>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..32768usize {
            let expected = array_u16[i] == target;
            let out_byte = slice[i / 8];
            let actual = (out_byte >> (i % 8)) & 1 == 1;
            assert_eq!(actual, expected, "u16 true i={i} target={target}");
        }
    }
}

#[test]
fn from_packed_u16_false_test() {
    let array_u16: [u16; 32768] = std::array::from_fn(|i| {
        ((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            % 512) as u16
    });
    let array: [u64; 8192] = unsafe { transmute(array_u16) };

    for target in [0u16, 7, 63, 255, 511] {
        let bitfield = Bitfield::from_packed_u16::<false>(&array, target);
        let slice_u32: [u32; 1024] = bitfield.as_array();
        let slice: &[u8; 4096] = unsafe { transmute(&slice_u32) };
        for i in 0..32768usize {
            let expected = array_u16[i] != target;
            let out_byte = slice[i / 8];
            let actual = (out_byte >> (i % 8)) & 1 == 1;
            assert_eq!(actual, expected, "u16 false i={i} target={target}");
        }
    }
}
