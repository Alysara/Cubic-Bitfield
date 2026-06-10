pub const NUM_LANES: usize = cfg_select! {
    // x86/x86_64
    target_feature = "avx512f" => 16,
    target_feature = "avx2"    => 8,
    target_feature = "avx"     => 8,
    target_feature = "sse2"    => 4,

    // AArch64 / ARM
    target_feature = "sve2"    => 16,
    target_feature = "sve"     => 16,
    target_feature = "neon"    => 4,

    // WASM
    target_feature = "simd128" => 4,

    // RISC-V
    target_feature = "v"       => 8,

    _                          => 4,
};
