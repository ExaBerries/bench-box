#include <gtest/gtest.h>
#include <bench_box/isa/intrin.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#if defined(BENCH_BOX_SIMD_REG4F) && defined(BENCH_BOX_SIMD_REG4I32)
TEST(Simd4F, ArithmeticBitwiseAndBlend) {
	alignas(alignof(bench_box::simd::reg4f)) float a_data[4] = {1.5f, -2.0f, 3.0f, -4.0f};
	alignas(alignof(bench_box::simd::reg4f)) float b_data[4] = {2.0f, 4.0f, -1.0f, 0.5f};
	alignas(alignof(bench_box::simd::reg4f)) float out[4];

	const bench_box::simd::reg4f a = bench_box::simd::load4f4f(a_data[0]);
	const bench_box::simd::reg4f b = bench_box::simd::load4f4f(b_data[0]);

	bench_box::simd::store4f(out[0], bench_box::simd::add4f(a, b));
	EXPECT_FLOAT_EQ(out[0], 3.5f);
	EXPECT_FLOAT_EQ(out[1], 2.0f);
	EXPECT_FLOAT_EQ(out[2], 2.0f);
	EXPECT_FLOAT_EQ(out[3], -3.5f);

	bench_box::simd::store4f(out[0], bench_box::simd::sub4f(a, b));
	EXPECT_FLOAT_EQ(out[0], -0.5f);
	EXPECT_FLOAT_EQ(out[1], -6.0f);
	EXPECT_FLOAT_EQ(out[2], 4.0f);
	EXPECT_FLOAT_EQ(out[3], -4.5f);

	bench_box::simd::store4f(out[0], bench_box::simd::mul4f(a, b));
	EXPECT_FLOAT_EQ(out[0], 3.0f);
	EXPECT_FLOAT_EQ(out[1], -8.0f);
	EXPECT_FLOAT_EQ(out[2], -3.0f);
	EXPECT_FLOAT_EQ(out[3], -2.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::div4f(a, b));
	EXPECT_FLOAT_EQ(out[0], 0.75f);
	EXPECT_FLOAT_EQ(out[1], -0.5f);
	EXPECT_FLOAT_EQ(out[2], -3.0f);
	EXPECT_FLOAT_EQ(out[3], -8.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::sqrt4f(bench_box::simd::abs4f(a)));
	EXPECT_FLOAT_EQ(out[0], std::sqrt(1.5f));
	EXPECT_FLOAT_EQ(out[1], std::sqrt(2.0f));
	EXPECT_FLOAT_EQ(out[2], std::sqrt(3.0f));
	EXPECT_FLOAT_EQ(out[3], 2.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::min4f(a, b));
	EXPECT_FLOAT_EQ(out[0], 1.5f);
	EXPECT_FLOAT_EQ(out[1], -2.0f);
	EXPECT_FLOAT_EQ(out[2], -1.0f);
	EXPECT_FLOAT_EQ(out[3], -4.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::max4f(a, b));
	EXPECT_FLOAT_EQ(out[0], 2.0f);
	EXPECT_FLOAT_EQ(out[1], 4.0f);
	EXPECT_FLOAT_EQ(out[2], 3.0f);
	EXPECT_FLOAT_EQ(out[3], 0.5f);

	bench_box::simd::store4f(out[0], bench_box::simd::neg4f(a));
	EXPECT_FLOAT_EQ(out[0], -1.5f);
	EXPECT_FLOAT_EQ(out[1], 2.0f);
	EXPECT_FLOAT_EQ(out[2], -3.0f);
	EXPECT_FLOAT_EQ(out[3], 4.0f);

	const bench_box::simd::reg4f all_bits = bench_box::simd::cast4i32_to_4f(bench_box::simd::create_mask_4i32<-1, -1, -1, -1>());
	bench_box::simd::store4f(out[0], bench_box::simd::and4f(a, all_bits));
	EXPECT_FLOAT_EQ(out[0], 1.5f);
	EXPECT_FLOAT_EQ(out[1], -2.0f);
	EXPECT_FLOAT_EQ(out[2], 3.0f);
	EXPECT_FLOAT_EQ(out[3], -4.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::andnot4f(all_bits, a));
	EXPECT_FLOAT_EQ(out[0], 0.0f);
	EXPECT_FLOAT_EQ(out[1], 0.0f);
	EXPECT_FLOAT_EQ(out[2], 0.0f);
	EXPECT_FLOAT_EQ(out[3], 0.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::or4f(bench_box::simd::zero_reg4f(), a));
	EXPECT_FLOAT_EQ(out[0], 1.5f);
	EXPECT_FLOAT_EQ(out[1], -2.0f);
	EXPECT_FLOAT_EQ(out[2], 3.0f);
	EXPECT_FLOAT_EQ(out[3], -4.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::xor4f(a, a));
	EXPECT_FLOAT_EQ(out[0], 0.0f);
	EXPECT_FLOAT_EQ(out[1], 0.0f);
	EXPECT_FLOAT_EQ(out[2], 0.0f);
	EXPECT_FLOAT_EQ(out[3], 0.0f);

	const bench_box::simd::reg4f mask = bench_box::simd::cmp_gt4f(a, b);
	bench_box::simd::store4f(out[0], bench_box::simd::blendv4f(a, b, mask));
	EXPECT_FLOAT_EQ(out[0], 1.5f);
	EXPECT_FLOAT_EQ(out[1], -2.0f);
	EXPECT_FLOAT_EQ(out[2], -1.0f);
	EXPECT_FLOAT_EQ(out[3], -4.0f);

	bench_box::simd::store4f(out[0], bench_box::simd::blend4f<0b1010>(a, b));
	EXPECT_FLOAT_EQ(out[0], 1.5f);
	EXPECT_FLOAT_EQ(out[1], 4.0f);
	EXPECT_FLOAT_EQ(out[2], 3.0f);
	EXPECT_FLOAT_EQ(out[3], 0.5f);
}

TEST(Simd4F, Compares) {
	alignas(alignof(bench_box::simd::reg4f)) float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	alignas(alignof(bench_box::simd::reg4f)) float b_data[4] = {2.0f, 2.0f, 1.0f, 4.0f};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t out[4];

	const bench_box::simd::reg4f a = bench_box::simd::load4f4f(a_data[0]);
	const bench_box::simd::reg4f b = bench_box::simd::load4f4f(b_data[0]);

	const auto store_mask = [&out](const bench_box::simd::reg4f& reg) {
		bench_box::simd::store4i32(out[0], bench_box::simd::cast4f_to_4i32(reg));
	};

	store_mask(bench_box::simd::cmp_lt4f(a, b));
	EXPECT_EQ(out[0], -1);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], 0);
	EXPECT_EQ(out[3], 0);

	store_mask(bench_box::simd::cmp_le4f(a, b));
	EXPECT_EQ(out[0], -1);
	EXPECT_EQ(out[1], -1);
	EXPECT_EQ(out[2], 0);
	EXPECT_EQ(out[3], -1);

	store_mask(bench_box::simd::cmp_gt4f(a, b));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], -1);
	EXPECT_EQ(out[3], 0);

	store_mask(bench_box::simd::cmp_ge4f(a, b));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], -1);
	EXPECT_EQ(out[2], -1);
	EXPECT_EQ(out[3], -1);

	store_mask(bench_box::simd::cmp_eq4f(a, b));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], -1);
	EXPECT_EQ(out[2], 0);
	EXPECT_EQ(out[3], -1);

	store_mask(bench_box::simd::cmp_neq4f(a, b));
	EXPECT_EQ(out[0], -1);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], -1);
	EXPECT_EQ(out[3], 0);
}

TEST(Simd4F, FloorAndConversions) {
	alignas(alignof(bench_box::simd::reg4f)) float data[4] = {1.9f, -1.1f, -0.5f, 3.0f};
	alignas(alignof(bench_box::simd::reg4f)) float floats[4];
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t ints[4];

	const bench_box::simd::reg4f values = bench_box::simd::load4f4f(data[0]);

	bench_box::simd::store4f(floats[0], bench_box::simd::floor4f(values));
	EXPECT_FLOAT_EQ(floats[0], 1.0f);
	EXPECT_FLOAT_EQ(floats[1], -2.0f);
	EXPECT_FLOAT_EQ(floats[2], -1.0f);
	EXPECT_FLOAT_EQ(floats[3], 3.0f);

	bench_box::simd::store4i32(ints[0], bench_box::simd::cvtt4f_to_4i32(values));
	EXPECT_EQ(ints[0], 1);
	EXPECT_EQ(ints[1], -1);
	EXPECT_EQ(ints[2], 0);
	EXPECT_EQ(ints[3], 3);

	const bench_box::simd::reg4f rounded_data = bench_box::simd::load1f4f(1.6f);
	bench_box::simd::store4i32(ints[0], bench_box::simd::cvt4f_to_4i32(rounded_data));
	EXPECT_EQ(ints[0], 2);
	EXPECT_EQ(ints[1], 2);
	EXPECT_EQ(ints[2], 2);
	EXPECT_EQ(ints[3], 2);

	bench_box::simd::store4f(floats[0], bench_box::simd::cvt4i32_to_4f(bench_box::simd::create_mask_4i32<1, -2, 3, -4>()));
	EXPECT_FLOAT_EQ(floats[0], 1.0f);
	EXPECT_FLOAT_EQ(floats[1], -2.0f);
	EXPECT_FLOAT_EQ(floats[2], 3.0f);
	EXPECT_FLOAT_EQ(floats[3], -4.0f);
}

TEST(Simd4F, UnalignedLoadStore) {
	float data[16] = {};
	for (int i = 0; i < 16; i++) {
		data[i] = static_cast<float>(i) + 0.25f;
	}

	const bench_box::simd::reg4f loaded = bench_box::simd::loadu4f4f(data[1]);
	bench_box::simd::storeu4f(data[5], loaded);
	EXPECT_FLOAT_EQ(data[5], 1.25f);
	EXPECT_FLOAT_EQ(data[6], 2.25f);
	EXPECT_FLOAT_EQ(data[7], 3.25f);
	EXPECT_FLOAT_EQ(data[8], 4.25f);
}

TEST(Simd4I32, ArithmeticLogicAndShifts) {
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t a_data[4] = {1, -2, 3, -4};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t b_data[4] = {5, 6, -7, 8};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t out[4];

	const bench_box::simd::reg4i32 a = bench_box::simd::load4i32(a_data[0]);
	const bench_box::simd::reg4i32 b = bench_box::simd::load4i32(b_data[0]);

	bench_box::simd::store4i32(out[0], bench_box::simd::add4i32(a, b));
	EXPECT_EQ(out[0], 6);
	EXPECT_EQ(out[1], 4);
	EXPECT_EQ(out[2], -4);
	EXPECT_EQ(out[3], 4);

	bench_box::simd::store4i32(out[0], bench_box::simd::sub4i32(a, b));
	EXPECT_EQ(out[0], -4);
	EXPECT_EQ(out[1], -8);
	EXPECT_EQ(out[2], 10);
	EXPECT_EQ(out[3], -12);

	bench_box::simd::store4i32(out[0], bench_box::simd::and4i32(a, b));
	EXPECT_EQ(out[0], 1);
	EXPECT_EQ(out[1], 6);
	EXPECT_EQ(out[2], 1);
	EXPECT_EQ(out[3], 8);

	bench_box::simd::store4i32(out[0], bench_box::simd::or4i32(a, b));
	EXPECT_EQ(out[0], 5);
	EXPECT_EQ(out[1], -2);
	EXPECT_EQ(out[2], -5);
	EXPECT_EQ(out[3], -4);

	bench_box::simd::store4i32(out[0], bench_box::simd::xor4i32(a, b));
	EXPECT_EQ(out[0], 4);
	EXPECT_EQ(out[1], -8);
	EXPECT_EQ(out[2], -6);
	EXPECT_EQ(out[3], -12);

	bench_box::simd::store4i32(out[0], bench_box::simd::andnot4i32(a, b));
	EXPECT_EQ(out[0], 4);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], -8);
	EXPECT_EQ(out[3], 0);

	bench_box::simd::store4i32(out[0], bench_box::simd::sl4i32<2>(a));
	EXPECT_EQ(out[0], 4);
	EXPECT_EQ(out[1], -8);
	EXPECT_EQ(out[2], 12);
	EXPECT_EQ(out[3], -16);

	bench_box::simd::store4i32(out[0], bench_box::simd::sr4i32<1>(a));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], 0x7FFFFFFF);
	EXPECT_EQ(out[2], 1);
	EXPECT_EQ(out[3], 0x7FFFFFFE);

	bench_box::simd::store4i32(out[0], bench_box::simd::sra4i32<1>(a));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], -1);
	EXPECT_EQ(out[2], 1);
	EXPECT_EQ(out[3], -2);
}

TEST(Simd4I32, MulMinMax) {
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t a_data[4] = {
		std::numeric_limits<std::int32_t>::min(),
		-3,
		123456,
		std::numeric_limits<std::int32_t>::max()
	};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t b_data[4] = {
		-1,
		7,
		-654321,
		2
	};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t out[4];

	const bench_box::simd::reg4i32 a = bench_box::simd::load4i32(a_data[0]);
	const bench_box::simd::reg4i32 b = bench_box::simd::load4i32(b_data[0]);

	bench_box::simd::store4i32(out[0], bench_box::simd::mul4i32(a, b));
	for (int lane = 0; lane < 4; lane++) {
		const auto expected = static_cast<std::int32_t>(
			static_cast<std::uint32_t>(a_data[lane]) * static_cast<std::uint32_t>(b_data[lane])
		);
		EXPECT_EQ(out[lane], expected);
	}

	bench_box::simd::store4i32(out[0], bench_box::simd::min4i32(a, b));
	EXPECT_EQ(out[0], std::numeric_limits<std::int32_t>::min());
	EXPECT_EQ(out[1], -3);
	EXPECT_EQ(out[2], -654321);
	EXPECT_EQ(out[3], 2);

	bench_box::simd::store4i32(out[0], bench_box::simd::max4i32(a, b));
	EXPECT_EQ(out[0], -1);
	EXPECT_EQ(out[1], 7);
	EXPECT_EQ(out[2], 123456);
	EXPECT_EQ(out[3], std::numeric_limits<std::int32_t>::max());

	std::mt19937 rng(0x5eed1234u);
	for (int iteration = 0; iteration < 1000; iteration++) {
		for (int lane = 0; lane < 4; lane++) {
			a_data[lane] = static_cast<std::int32_t>(rng());
			b_data[lane] = static_cast<std::int32_t>(rng());
		}
		const bench_box::simd::reg4i32 random_a = bench_box::simd::load4i32(a_data[0]);
		const bench_box::simd::reg4i32 random_b = bench_box::simd::load4i32(b_data[0]);

		bench_box::simd::store4i32(out[0], bench_box::simd::mul4i32(random_a, random_b));
		for (int lane = 0; lane < 4; lane++) {
			const auto expected = static_cast<std::int32_t>(
				static_cast<std::uint32_t>(a_data[lane]) * static_cast<std::uint32_t>(b_data[lane])
			);
			EXPECT_EQ(out[lane], expected);
		}

		bench_box::simd::store4i32(out[0], bench_box::simd::min4i32(random_a, random_b));
		for (int lane = 0; lane < 4; lane++) {
			EXPECT_EQ(out[lane], a_data[lane] < b_data[lane] ? a_data[lane] : b_data[lane]);
		}

		bench_box::simd::store4i32(out[0], bench_box::simd::max4i32(random_a, random_b));
		for (int lane = 0; lane < 4; lane++) {
			EXPECT_EQ(out[lane], a_data[lane] > b_data[lane] ? a_data[lane] : b_data[lane]);
		}
	}
}

TEST(Simd4I32, UnsignedMinMax) {
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t a_data[4] = {
		static_cast<std::int32_t>(0x80000000u),
		static_cast<std::int32_t>(0xFFFFFFFFu),
		1,
		0x7FFFFFFF
	};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t b_data[4] = {
		static_cast<std::int32_t>(0x7FFFFFFFu),
		static_cast<std::int32_t>(0x00000001u),
		static_cast<std::int32_t>(0xFFFFFFFFu),
		0
	};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t out[4];

	const bench_box::simd::reg4i32 a = bench_box::simd::load4i32(a_data[0]);
	const bench_box::simd::reg4i32 b = bench_box::simd::load4i32(b_data[0]);

	bench_box::simd::store4i32(out[0], bench_box::simd::minu4i32(a, b));
	for (int lane = 0; lane < 4; lane++) {
		const auto expected = static_cast<std::uint32_t>(a_data[lane]) < static_cast<std::uint32_t>(b_data[lane])
			? a_data[lane]
			: b_data[lane];
		EXPECT_EQ(out[lane], expected);
	}

	bench_box::simd::store4i32(out[0], bench_box::simd::maxu4i32(a, b));
	for (int lane = 0; lane < 4; lane++) {
		const auto expected = static_cast<std::uint32_t>(a_data[lane]) > static_cast<std::uint32_t>(b_data[lane])
			? a_data[lane]
			: b_data[lane];
		EXPECT_EQ(out[lane], expected);
	}
}

TEST(Simd4I32, ComparesAndBlend) {
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t a_data[4] = {1, 2, 3, 4};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t b_data[4] = {2, 2, 1, 4};
	alignas(alignof(bench_box::simd::reg4i32)) std::int32_t out[4];

	const bench_box::simd::reg4i32 a = bench_box::simd::load4i32(a_data[0]);
	const bench_box::simd::reg4i32 b = bench_box::simd::load4i32(b_data[0]);

	bench_box::simd::store4i32(out[0], bench_box::simd::cmp_lt4i32(a, b));
	EXPECT_EQ(out[0], -1);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], 0);
	EXPECT_EQ(out[3], 0);

	bench_box::simd::store4i32(out[0], bench_box::simd::cmp_gt4i32(a, b));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], 0);
	EXPECT_EQ(out[2], -1);
	EXPECT_EQ(out[3], 0);

	bench_box::simd::store4i32(out[0], bench_box::simd::cmp_eq4i32(a, b));
	EXPECT_EQ(out[0], 0);
	EXPECT_EQ(out[1], -1);
	EXPECT_EQ(out[2], 0);
	EXPECT_EQ(out[3], -1);

	const bench_box::simd::reg4i32 mask = bench_box::simd::cmp_gt4i32(a, b);
	bench_box::simd::store4i32(out[0], bench_box::simd::blendv4i32(a, b, mask));
	EXPECT_EQ(out[0], 1);
	EXPECT_EQ(out[1], 2);
	EXPECT_EQ(out[2], 1);
	EXPECT_EQ(out[3], 4);
}

TEST(Simd4I32, UnalignedLoadStoreAndBitcast) {
	std::int32_t data[16] = {};
	for (int i = 0; i < 16; i++) {
		data[i] = i * 3 - 7;
	}

	const bench_box::simd::reg4i32 loaded = bench_box::simd::loadu4i32(data[1]);
	bench_box::simd::storeu4i32(data[5], loaded);
	EXPECT_EQ(data[5], -4);
	EXPECT_EQ(data[6], -1);
	EXPECT_EQ(data[7], 2);
	EXPECT_EQ(data[8], 5);

	const bench_box::simd::reg4f bits = bench_box::simd::cast4i32_to_4f(bench_box::simd::create_mask_4i32<0x3F800000, 0x40000000, 0x40400000, 0x40800000>());
	EXPECT_FLOAT_EQ(bench_box::simd::extract4f_lo(bits), 1.0f);

	alignas(alignof(bench_box::simd::reg4f)) float floats[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	bench_box::simd::store4i32(data[0], bench_box::simd::cast4f_to_4i32(bench_box::simd::load4f4f(floats[0])));
	EXPECT_EQ(data[0], 0x3F800000);
	EXPECT_EQ(data[1], 0x40000000);
	EXPECT_EQ(data[2], 0x40400000);
	EXPECT_EQ(data[3], 0x40800000);
}
#endif

#if defined(BENCH_BOX_SIMD_REG8F) && defined(BENCH_BOX_SIMD_REG8I32)
TEST(Simd8, UnalignedFloorCastAndBitcast) {
	float data[24] = {};
	for (int i = 0; i < 24; i++) {
		data[i] = static_cast<float>(i) + 0.75f;
	}

	const bench_box::simd::reg8f loaded = bench_box::simd::loadu8f8f(data[1]);
	bench_box::simd::storeu8f(data[9], loaded);
	for (int lane = 0; lane < 8; lane++) {
		EXPECT_FLOAT_EQ(data[9 + lane], static_cast<float>(1 + lane) + 0.75f);
	}

	bench_box::simd::storeu8f(data[1], bench_box::simd::floor8f(loaded));
	for (int lane = 0; lane < 8; lane++) {
		EXPECT_FLOAT_EQ(data[1 + lane], static_cast<float>(1 + lane));
	}

	const bench_box::simd::reg8i32 bits = bench_box::simd::cast8f_to_8i32(loaded);
	alignas(alignof(bench_box::simd::reg8f)) float cast_back[8];
	bench_box::simd::store8f(cast_back[0], bench_box::simd::cast8i32_to_8f(bits));
	for (int lane = 0; lane < 8; lane++) {
		EXPECT_FLOAT_EQ(cast_back[lane], static_cast<float>(1 + lane) + 0.75f);
	}

	std::int32_t ints[16] = {};
	for (int i = 0; i < 16; i++) {
		ints[i] = i - 5;
	}
	const bench_box::simd::reg8i32 loaded_ints = bench_box::simd::loadu8i32(ints[1]);
	bench_box::simd::storeu8i32(ints[5], loaded_ints);
	for (int lane = 0; lane < 8; lane++) {
		EXPECT_EQ(ints[5 + lane], -4 + lane);
	}
}
#endif

#if defined(BENCH_BOX_SIMD_REG16F) && defined(BENCH_BOX_SIMD_REG16I32)
TEST(Simd16, UnalignedFloorCastAndMasks) {
	float data[40] = {};
	for (int i = 0; i < 40; i++) {
		data[i] = static_cast<float>(i) + 0.5f;
	}

	const bench_box::simd::reg16f loaded = bench_box::simd::loadu16f16f(data[3]);
	bench_box::simd::storeu16f(data[20], loaded);
	for (int lane = 0; lane < 16; lane++) {
		EXPECT_FLOAT_EQ(data[20 + lane], static_cast<float>(3 + lane) + 0.5f);
	}

	bench_box::simd::storeu16f(data[3], bench_box::simd::floor16f(loaded));
	for (int lane = 0; lane < 16; lane++) {
		EXPECT_FLOAT_EQ(data[3 + lane], static_cast<float>(3 + lane));
	}

	const bench_box::simd::reg16i32 bits = bench_box::simd::cast16f_to_16i32(loaded);
	EXPECT_FLOAT_EQ(bench_box::simd::extract16f_lo(bench_box::simd::cast16i32_to_16f(bits)), 3.5f);

	std::int32_t ints[32] = {};
	for (int i = 0; i < 32; i++) {
		ints[i] = i - 10;
	}
	const bench_box::simd::reg16i32 loaded_ints = bench_box::simd::loadu16i32(ints[4]);
	bench_box::simd::storeu16i32(ints[12], loaded_ints);
	for (int lane = 0; lane < 16; lane++) {
		EXPECT_EQ(ints[12 + lane], -6 + lane);
	}

	const bench_box::simd::mask16 a = static_cast<bench_box::simd::mask16>(0x0F0F);
	const bench_box::simd::mask16 b = static_cast<bench_box::simd::mask16>(0x00FF);
	EXPECT_EQ(bench_box::simd::and16mask(a, b), static_cast<bench_box::simd::mask16>(0x000F));
	EXPECT_EQ(bench_box::simd::or16mask(a, b), static_cast<bench_box::simd::mask16>(0x0FFF));
	EXPECT_EQ(bench_box::simd::not16mask(a), static_cast<bench_box::simd::mask16>(0xF0F0));
}
#endif
