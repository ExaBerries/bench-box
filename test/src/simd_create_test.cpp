#include <gtest/gtest.h>
#include <bench_box/isa/intrin.h>

#include <cstdint>

#if defined(BENCH_BOX_SIMD_SSE2)
TEST(SimdCreate, Sse2) {
	alignas(alignof(bench_box::simd::reg4f)) float floats[4] = {};
	bench_box::simd::store4f(floats[0], bench_box::simd::create4f(1.0f, 2.0f, 3.0f, 4.0f));
	EXPECT_FLOAT_EQ(floats[0], 1.0f);
	EXPECT_FLOAT_EQ(floats[1], 2.0f);
	EXPECT_FLOAT_EQ(floats[2], 3.0f);
	EXPECT_FLOAT_EQ(floats[3], 4.0f);

	alignas(alignof(bench_box::simd::reg2d)) double doubles[2] = {};
	bench_box::simd::store2d(doubles[0], bench_box::simd::create2d(5.0, 6.0));
	EXPECT_DOUBLE_EQ(doubles[0], 5.0);
	EXPECT_DOUBLE_EQ(doubles[1], 6.0);

	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t ints[4] = {};
	bench_box::simd::store4i32(ints[0], bench_box::simd::create4i32(7, 8, 9, 10));
	EXPECT_EQ(ints[0], 7);
	EXPECT_EQ(ints[1], 8);
	EXPECT_EQ(ints[2], 9);
	EXPECT_EQ(ints[3], 10);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX)
TEST(SimdCreate, Avx) {
	alignas(alignof(bench_box::simd::reg8f)) float floats[8] = {};
	bench_box::simd::store8f(floats[0], bench_box::simd::create8f(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f));
	for (int i = 0; i < 8; i++) {
		EXPECT_FLOAT_EQ(floats[i], static_cast<float>(i + 1));
	}

	alignas(alignof(bench_box::simd::reg4d)) double doubles[4] = {};
	bench_box::simd::store4d(doubles[0], bench_box::simd::create4d(9.0, 10.0, 11.0, 12.0));
	for (int i = 0; i < 4; i++) {
		EXPECT_DOUBLE_EQ(doubles[i], static_cast<double>(i + 9));
	}

	alignas(alignof(bench_box::simd::reg2i64)) std::int64_t ints[2] = {};
	bench_box::simd::store2i64(ints[0], bench_box::simd::create2i64(13, 14));
	EXPECT_EQ(ints[0], 13);
	EXPECT_EQ(ints[1], 14);
}
#endif

#if defined(BENCH_BOX_SIMD_AVX2)
TEST(SimdCreate, Avx2) {
	alignas(alignof(bench_box::simd::reg8i32)) std::int32_t ints32[8] = {};
	bench_box::simd::store8i32(ints32[0], bench_box::simd::create8i32(1, 2, 3, 4, 5, 6, 7, 8));
	for (int i = 0; i < 8; i++) {
		EXPECT_EQ(ints32[i], i + 1);
	}

	alignas(alignof(bench_box::simd::reg4i64)) std::int64_t ints64[4] = {};
	bench_box::simd::store4i64(ints64[0], bench_box::simd::create4i64(9, 10, 11, 12));
	for (int i = 0; i < 4; i++) {
		EXPECT_EQ(ints64[i], i + 9);
	}
}
#endif

#if defined(BENCH_BOX_SIMD_AVX512F)
TEST(SimdCreate, Avx512) {
	alignas(alignof(bench_box::simd::reg16f)) float floats[16] = {};
	bench_box::simd::store16f(floats[0], bench_box::simd::create16f(
		1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
	));
	for (int i = 0; i < 16; i++) {
		EXPECT_FLOAT_EQ(floats[i], static_cast<float>(i + 1));
	}

	alignas(alignof(bench_box::simd::reg8d)) double doubles[8] = {};
	bench_box::simd::store8d(doubles[0], bench_box::simd::create8d(17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0));
	for (int i = 0; i < 8; i++) {
		EXPECT_DOUBLE_EQ(doubles[i], static_cast<double>(i + 17));
	}

	alignas(alignof(bench_box::simd::reg16i32)) std::int32_t ints32[16] = {};
	bench_box::simd::store16i32(ints32[0], bench_box::simd::create16i32(
		25, 26, 27, 28, 29, 30, 31, 32,
		33, 34, 35, 36, 37, 38, 39, 40
	));
	for (int i = 0; i < 16; i++) {
		EXPECT_EQ(ints32[i], i + 25);
	}

	alignas(alignof(bench_box::simd::reg8i64)) std::int64_t ints64[8] = {};
	bench_box::simd::store8i64(ints64[0], bench_box::simd::create8i64(41, 42, 43, 44, 45, 46, 47, 48));
	for (int i = 0; i < 8; i++) {
		EXPECT_EQ(ints64[i], i + 41);
	}
}
#endif
