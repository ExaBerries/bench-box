#include <gtest/gtest.h>
#include <bench_box/isa/intrin.h>

#include <cstdint>

#if defined(BENCH_BOX_SIMD_SSE2)
TEST(SimdMask, Sse2RoundTrip) {
	EXPECT_EQ(bench_box::simd::maskbits4f(bench_box::simd::mask_from_bits4f(0xAu)), 0xAu);
	EXPECT_EQ(bench_box::simd::maskbits2d(bench_box::simd::mask_from_bits2d(0x3u)), 0x3u);
	EXPECT_EQ(bench_box::simd::maskbits4i32(bench_box::simd::mask_from_bits4i32(0xFu)), 0xFu);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX)
TEST(SimdMask, AvxRoundTrip) {
	EXPECT_EQ(bench_box::simd::maskbits8f(bench_box::simd::mask_from_bits8f(0xA5u)), 0xA5u);
	EXPECT_EQ(bench_box::simd::maskbits4d(bench_box::simd::mask_from_bits4d(0xFu)), 0xFu);
	EXPECT_EQ(bench_box::simd::maskbits2i64(bench_box::simd::mask_from_bits2i64(0x3u)), 0x3u);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX2)
TEST(SimdMask, Avx2RoundTrip) {
	EXPECT_EQ(bench_box::simd::maskbits8i32(bench_box::simd::mask_from_bits8i32(0xA5u)), 0xA5u);
	EXPECT_EQ(bench_box::simd::maskbits4i64(bench_box::simd::mask_from_bits4i64(0xFu)), 0xFu);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512F)
TEST(SimdMask, Avx512KRoundTrip) {
	EXPECT_EQ(bench_box::simd::maskbits16f(bench_box::simd::mask_from_bits16f(0xA5A5u)), 0xA5A5u);
	EXPECT_EQ(bench_box::simd::maskbits8d(bench_box::simd::mask_from_bits8d(0xA5u)), 0xA5u);
	EXPECT_EQ(bench_box::simd::maskbits16i32(bench_box::simd::mask_from_bits16i32(0x5A5Au)), 0x5A5Au);
	EXPECT_EQ(bench_box::simd::maskbits8i64(bench_box::simd::mask_from_bits8i64(0x5Au)), 0x5Au);
}

TEST(SimdCompress, Avx512) {
	alignas(64) const std::int32_t input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	const bench_box::simd::reg16i32 values = bench_box::simd::load16i32(input[0]);
	const bench_box::simd::mask16 mask = bench_box::simd::mask_from_bits16i32(0x5555u);
	alignas(64) std::int32_t output[16] = {};
	bench_box::simd::store16i32(output[0], bench_box::simd::maskz_compress16i32(values, mask));
	for (int i = 0; i < 8; i++) {
		EXPECT_EQ(output[i], i * 2);
	}
	for (int i = 8; i < 16; i++) {
		EXPECT_EQ(output[i], 0);
	}

	alignas(64) std::int64_t input64[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	const bench_box::simd::reg8i64 values64 = bench_box::simd::load8i64(input64[0]);
	const bench_box::simd::mask8 mask64 = bench_box::simd::mask_from_bits8i64(0x55u);
	alignas(64) std::int64_t output64[8] = {};
	bench_box::simd::mask_compressstoreu8i64(output64[0], mask64, values64);
	for (int i = 0; i < 4; i++) {
		EXPECT_EQ(output64[i], i * 2);
	}
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512VL)
TEST(SimdCompress, Vl) {
	alignas(32) const std::int32_t input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	const bench_box::simd::reg8i32 values = bench_box::simd::load8i32(input[0]);
	const bench_box::simd::mask8 mask = static_cast<bench_box::simd::mask8>(0xAAu);
	alignas(32) std::int32_t output[8] = {};
	bench_box::simd::store8i32(output[0], bench_box::simd::maskz_compress8i32(values, mask));
	EXPECT_EQ(output[0], 1);
	EXPECT_EQ(output[1], 3);
	EXPECT_EQ(output[2], 5);
	EXPECT_EQ(output[3], 7);
	EXPECT_EQ(output[4], 0);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512CD)
TEST(SimdConflict, Avx512) {
	const bench_box::simd::reg16i32 values = bench_box::simd::create16i32(
		0, 0, 1, 0, 1, 2, 0, 1,
		2, 3, 0, 1, 2, 3, 4, 5
	);
	alignas(64) std::int32_t output[16] = {};
	bench_box::simd::store16i32(output[0], bench_box::simd::conflict16i32(values));
	EXPECT_EQ(output[0], 0);
	EXPECT_EQ(output[1], 0x1);
	EXPECT_EQ(output[2], 0);
	EXPECT_EQ(output[3], 0x3);
}
#endif

TEST(SimdMisc, Prefetch) {
	int value = 0;
	bench_box::prefetch_read(&value);
	bench_box::prefetch_write(&value);
	bench_box::prefetch_nta(&value);
	SUCCEED();
}
