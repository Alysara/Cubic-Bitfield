pub fn gen_rand_packed<const N: usize>() -> [u64; N] {
    let mut rand: u64 = 0x9E3779B97F4A7C15;
    std::array::from_fn(|_| {
        rand = rand.wrapping_mul(0xBF58476D1CE4E5B9);
        rand
    })
}

pub fn gen_alternating_value<const N: usize>(val1: u64, val2: u64) -> [u64; N] {
    let mut counter = 0;
    std::array::from_fn(|_| {
        counter += 1;
        if (counter % 2) == 0 { val1 } else { val2 }
    })
}
