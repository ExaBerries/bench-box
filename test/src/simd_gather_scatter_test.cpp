#include <gtest/gtest.h>
#include <bench_box/isa/intrin.h>

#include <cstdint>

#if defined(BENCH_BOX_SIMD_AVX2)
TEST(SimdGather, GatherFourLanes) {
	const float floats[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
	const double doubles[8] = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0};
	const std::int32_t ints32[8] = {30, 31, 32, 33, 34, 35, 36, 37};
	const std::int64_t ints64[8] = {40, 41, 42, 43, 44, 45, 46, 47};
	const bench_box::simd::reg4i64 indices = bench_box::simd::create4i64(1, 3, 5, 7);

	alignas(16) float float_out[4] = {};
	bench_box::simd::store4f(float_out[0], bench_box::simd::gather4f4i64(floats, indices));
	EXPECT_FLOAT_EQ(float_out[0], 11.0f);
	EXPECT_FLOAT_EQ(float_out[1], 13.0f);
	EXPECT_FLOAT_EQ(float_out[2], 15.0f);
	EXPECT_FLOAT_EQ(float_out[3], 17.0f);

	alignas(32) double double_out[4] = {};
	bench_box::simd::store4d(double_out[0], bench_box::simd::gather4d4i64(doubles, indices));
	EXPECT_DOUBLE_EQ(double_out[0], 21.0);
	EXPECT_DOUBLE_EQ(double_out[1], 23.0);
	EXPECT_DOUBLE_EQ(double_out[2], 25.0);
	EXPECT_DOUBLE_EQ(double_out[3], 27.0);

	alignas(16) std::int32_t int32_out[4] = {};
	bench_box::simd::store4i32(int32_out[0], bench_box::simd::gather4i324i64(ints32, indices));
	EXPECT_EQ(int32_out[0], 31);
	EXPECT_EQ(int32_out[1], 33);
	EXPECT_EQ(int32_out[2], 35);
	EXPECT_EQ(int32_out[3], 37);

	alignas(32) std::int64_t int64_out[4] = {};
	bench_box::simd::store4i64(int64_out[0], bench_box::simd::gather4i644i64(ints64, indices));
	EXPECT_EQ(int64_out[0], 41);
	EXPECT_EQ(int64_out[1], 43);
	EXPECT_EQ(int64_out[2], 45);
	EXPECT_EQ(int64_out[3], 47);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX2) && defined(BENCH_BOX_SIMD_AVX512F) && defined(BENCH_BOX_SIMD_AVX512VL)
TEST(SimdScatter, ScatterFourLanes) {
	const bench_box::simd::reg4i64 indices = bench_box::simd::create4i64(1, 3, 5, 7);
	const bench_box::simd::mask8 mask = static_cast<bench_box::simd::mask8>(0x0F);

	alignas(16) const float float_in[4] = {10.0f, 11.0f, 12.0f, 13.0f};
	alignas(32) float float_out[8] = {};
	bench_box::simd::scatter4f4i64(float_out, mask, indices, bench_box::simd::load4f4f(float_in[0]));
	EXPECT_FLOAT_EQ(float_out[1], 10.0f);
	EXPECT_FLOAT_EQ(float_out[3], 11.0f);
	EXPECT_FLOAT_EQ(float_out[5], 12.0f);
	EXPECT_FLOAT_EQ(float_out[7], 13.0f);

	alignas(32) const double double_in[4] = {20.0, 21.0, 22.0, 23.0};
	alignas(32) double double_out[8] = {};
	bench_box::simd::scatter4d4i64(double_out, mask, indices, bench_box::simd::load4d4d(double_in[0]));
	EXPECT_DOUBLE_EQ(double_out[1], 20.0);
	EXPECT_DOUBLE_EQ(double_out[3], 21.0);
	EXPECT_DOUBLE_EQ(double_out[5], 22.0);
	EXPECT_DOUBLE_EQ(double_out[7], 23.0);

	alignas(16) const std::int32_t int32_in[4] = {30, 31, 32, 33};
	alignas(32) std::int32_t int32_out[8] = {};
	bench_box::simd::scatter4i324i64(int32_out, mask, indices, bench_box::simd::load4i32(int32_in[0]));
	EXPECT_EQ(int32_out[1], 30);
	EXPECT_EQ(int32_out[3], 31);
	EXPECT_EQ(int32_out[5], 32);
	EXPECT_EQ(int32_out[7], 33);

	alignas(32) const std::int64_t int64_in[4] = {40, 41, 42, 43};
	alignas(32) std::int64_t int64_out[8] = {};
	bench_box::simd::scatter4i644i64(int64_out, mask, indices, bench_box::simd::load4i64(int64_in[0]));
	EXPECT_EQ(int64_out[1], 40);
	EXPECT_EQ(int64_out[3], 41);
	EXPECT_EQ(int64_out[5], 42);
	EXPECT_EQ(int64_out[7], 43);
}

TEST(SimdScatter, MaskedScatter) {
	const bench_box::simd::reg4i64 indices = bench_box::simd::create4i64(0, 1, 2, 3);
	const bench_box::simd::mask8 mask = static_cast<bench_box::simd::mask8>(0x05);

	alignas(16) const float float_in[4] = {10.0f, 11.0f, 12.0f, 13.0f};
	alignas(16) float float_out[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	bench_box::simd::scatter4f4i64(float_out, mask, indices, bench_box::simd::load4f4f(float_in[0]));
	EXPECT_FLOAT_EQ(float_out[0], 10.0f);
	EXPECT_FLOAT_EQ(float_out[1], 2.0f);
	EXPECT_FLOAT_EQ(float_out[2], 12.0f);
	EXPECT_FLOAT_EQ(float_out[3], 4.0f);
}
#endif

#if defined(BENCH_BOX_SIMD_SSE2)
TEST(SimdGather, Sse2FallbackI32Indices) {
	const float floats[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
	const double doubles[8] = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0};
	const std::int32_t ints[8] = {30, 31, 32, 33, 34, 35, 36, 37};
	const bench_box::simd::reg4i32 indices = bench_box::simd::create4i32(7, 5, 3, 1);

	alignas(16) float floats_out[4] = {};
	bench_box::simd::store4f(floats_out[0], bench_box::simd::gather4f4i32(floats, indices));
	EXPECT_FLOAT_EQ(floats_out[0], 17.0f);
	EXPECT_FLOAT_EQ(floats_out[1], 15.0f);
	EXPECT_FLOAT_EQ(floats_out[3], 11.0f);

	alignas(16) double doubles_out[2] = {};
	bench_box::simd::store2d(doubles_out[0], bench_box::simd::gather2d4i32(doubles, indices));
	EXPECT_DOUBLE_EQ(doubles_out[0], 27.0);
	EXPECT_DOUBLE_EQ(doubles_out[1], 25.0);

	alignas(16) std::int32_t ints_out[4] = {};
	bench_box::simd::store4i32(ints_out[0], bench_box::simd::gather4i324i32(ints, indices));
	EXPECT_EQ(ints_out[0], 37);
	EXPECT_EQ(ints_out[1], 35);
	EXPECT_EQ(ints_out[3], 31);

	const bench_box::simd::reg4f mask = bench_box::simd::mask_from_bits4f(0x5u);
	alignas(16) float masked_out[4] = {};
	bench_box::simd::store4f(masked_out[0], bench_box::simd::maskz_gather4f4i32(floats, indices, mask));
	EXPECT_FLOAT_EQ(masked_out[0], 17.0f);
	EXPECT_FLOAT_EQ(masked_out[1], 0.0f);
	EXPECT_FLOAT_EQ(masked_out[2], 13.0f);
	EXPECT_FLOAT_EQ(masked_out[3], 0.0f);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX2)
TEST(SimdGather, I32Indices) {
	const float floats[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
	const double doubles[8] = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0};
	const std::int32_t ints32[8] = {30, 31, 32, 33, 34, 35, 36, 37};
	const std::int64_t ints64[8] = {40, 41, 42, 43, 44, 45, 46, 47};
	const bench_box::simd::reg8i32 indices8 = bench_box::simd::create8i32(7, 5, 3, 1, 6, 4, 2, 0);
	const bench_box::simd::reg4i32 indices4 = bench_box::simd::create4i32(3, 1, 7, 5);

	alignas(32) float floats_out[8] = {};
	bench_box::simd::store8f(floats_out[0], bench_box::simd::gather8f8i32(floats, indices8));
	EXPECT_FLOAT_EQ(floats_out[0], 17.0f);
	EXPECT_FLOAT_EQ(floats_out[1], 15.0f);
	EXPECT_FLOAT_EQ(floats_out[7], 10.0f);

	alignas(32) double doubles_out[4] = {};
	bench_box::simd::store4d(doubles_out[0], bench_box::simd::gather4d4i32(doubles, indices4));
	EXPECT_DOUBLE_EQ(doubles_out[0], 23.0);
	EXPECT_DOUBLE_EQ(doubles_out[1], 21.0);
	EXPECT_DOUBLE_EQ(doubles_out[3], 25.0);

	alignas(32) std::int32_t ints32_out[8] = {};
	bench_box::simd::store8i32(ints32_out[0], bench_box::simd::gather8i328i32(ints32, indices8));
	EXPECT_EQ(ints32_out[0], 37);
	EXPECT_EQ(ints32_out[1], 35);
	EXPECT_EQ(ints32_out[7], 30);

	alignas(32) std::int64_t ints64_out[4] = {};
	bench_box::simd::store4i64(ints64_out[0], bench_box::simd::gather4i644i32(ints64, indices4));
	EXPECT_EQ(ints64_out[0], 43);
	EXPECT_EQ(ints64_out[1], 41);
	EXPECT_EQ(ints64_out[3], 45);
}

TEST(SimdGather, MaskedI32Indices) {
	const float floats[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
	const bench_box::simd::reg8i32 indices = bench_box::simd::create8i32(0, 1, 2, 3, 4, 5, 6, 7);
	const bench_box::simd::reg8f mask = bench_box::simd::mask_from_bits8f(0xAA);
	const bench_box::simd::reg8f source = bench_box::simd::create8f(100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f);

	alignas(32) float merged[8] = {};
	bench_box::simd::store8f(merged[0], bench_box::simd::mask_gather8f8i32(source, floats, indices, mask));
	EXPECT_FLOAT_EQ(merged[0], 100.0f);
	EXPECT_FLOAT_EQ(merged[1], 11.0f);
	EXPECT_FLOAT_EQ(merged[2], 102.0f);
	EXPECT_FLOAT_EQ(merged[7], 17.0f);

	alignas(32) float zeroed[8] = {};
	bench_box::simd::store8f(zeroed[0], bench_box::simd::maskz_gather8f8i32(floats, indices, mask));
	EXPECT_FLOAT_EQ(zeroed[0], 0.0f);
	EXPECT_FLOAT_EQ(zeroed[1], 11.0f);
	EXPECT_FLOAT_EQ(zeroed[2], 0.0f);
	EXPECT_FLOAT_EQ(zeroed[7], 17.0f);
}

TEST(SimdScatter, Avx2FallbackI32Indices) {
	const bench_box::simd::reg8i32 indices = bench_box::simd::create8i32(7, 6, 5, 4, 3, 2, 1, 0);
	const bench_box::simd::reg8f mask = bench_box::simd::mask_from_bits8f(0x55);
	const bench_box::simd::reg8f values = bench_box::simd::create8f(10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f);

	alignas(32) float floats[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
	bench_box::simd::scatter8f8i32(floats, mask, indices, values);
	EXPECT_FLOAT_EQ(floats[7], 10.0f);
	EXPECT_FLOAT_EQ(floats[5], 12.0f);
	EXPECT_FLOAT_EQ(floats[3], 14.0f);
	EXPECT_FLOAT_EQ(floats[1], 16.0f);
	EXPECT_FLOAT_EQ(floats[0], -1.0f);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512VL)
TEST(SimdGather, VlMaskedI32Indices) {
	const float floats[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
	const bench_box::simd::reg8i32 indices = bench_box::simd::create8i32(0, 1, 2, 3, 4, 5, 6, 7);
	const bench_box::simd::mask8 mask = static_cast<bench_box::simd::mask8>(0xAA);
	const bench_box::simd::reg8f source = bench_box::simd::create8f(100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f);

	alignas(32) float merged[8] = {};
	bench_box::simd::store8f(merged[0], bench_box::simd::mask_gather8f8i32(source, floats, indices, mask));
	EXPECT_FLOAT_EQ(merged[0], 100.0f);
	EXPECT_FLOAT_EQ(merged[1], 11.0f);
	EXPECT_FLOAT_EQ(merged[7], 17.0f);
}

TEST(SimdScatter, VlI32Indices) {
	const bench_box::simd::reg8i32 indices = bench_box::simd::create8i32(7, 6, 5, 4, 3, 2, 1, 0);
	const bench_box::simd::mask8 mask = static_cast<bench_box::simd::mask8>(0x55);
	const bench_box::simd::reg8f values = bench_box::simd::create8f(10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f);

	alignas(32) float floats[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
	bench_box::simd::scatter8f8i32(floats, mask, indices, values);
	EXPECT_FLOAT_EQ(floats[7], 10.0f);
	EXPECT_FLOAT_EQ(floats[5], 12.0f);
	EXPECT_FLOAT_EQ(floats[3], 14.0f);
	EXPECT_FLOAT_EQ(floats[1], 16.0f);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512F)
TEST(SimdGather, Avx512I32Indices) {
	const float floats[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
	const bench_box::simd::reg16i32 indices = bench_box::simd::create16i32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	const bench_box::simd::mask16 mask = bench_box::simd::mask_from_bits16i32(0xAAAA);
	const bench_box::simd::reg16f source = bench_box::simd::create16f(
		100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f,
		108.0f, 109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f
	);

	alignas(64) float merged[16] = {};
	bench_box::simd::store16f(merged[0], bench_box::simd::mask_gather16f16i32(source, floats, indices, mask));
	EXPECT_FLOAT_EQ(merged[0], 100.0f);
	EXPECT_FLOAT_EQ(merged[1], 14.0f);
	EXPECT_FLOAT_EQ(merged[2], 102.0f);
	EXPECT_FLOAT_EQ(merged[15], 0.0f);
}

TEST(SimdScatter, Avx512I32Indices) {
	const bench_box::simd::reg16i32 indices = bench_box::simd::create16i32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	const bench_box::simd::mask16 mask = bench_box::simd::mask_from_bits16i32(0xAAAA);
	const bench_box::simd::reg16f values = bench_box::simd::create16f(
		10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
		18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f
	);

	alignas(64) float floats[16] = {};
	bench_box::simd::scatter16f16i32(floats, mask, indices, values);
	EXPECT_FLOAT_EQ(floats[0], 25.0f);
	EXPECT_FLOAT_EQ(floats[14], 11.0f);
	EXPECT_FLOAT_EQ(floats[12], 13.0f);
	EXPECT_FLOAT_EQ(floats[1], 0.0f);
}
#endif
