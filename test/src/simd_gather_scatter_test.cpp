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
