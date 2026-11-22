#include <gtest/gtest.h>
#include <bench_box/isa/intrin.h>
#include <array>
#include <cstdint>
#include <random>

namespace {
	constexpr std::array<std::uint64_t, 12> edge_values = {
		0ull,
		1ull,
		2ull,
		0x00000000FFFFFFFFull,
		0x0000000100000000ull,
		0xFFFFFFFFFFFFFFFFull,
		0x8000000000000000ull,
		0x7FFFFFFFFFFFFFFFull,
		0x123456789ABCDEF0ull,
		0xFFFFFFFF00000000ull,
		0x0000000100000001ull,
		0xFEDCBA9876543210ull
	};
} // namespace

#if defined(BENCH_BOX_SIMD_REG2I64)
TEST(SimdI64, Mul2) {
	for (const std::uint64_t a : edge_values) {
		for (const std::uint64_t b : edge_values) {
			alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_a[2] = {static_cast<std::int64_t>(a), static_cast<std::int64_t>(a)};
			alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_b[2] = {static_cast<std::int64_t>(b), static_cast<std::int64_t>(b)};
			alignas(alignof(bench_box::simd::reg2i64)) std::int64_t out[2];

			const bench_box::simd::reg2i64 va = bench_box::simd::load2i64(in_a[0]);
			const bench_box::simd::reg2i64 vb = bench_box::simd::load2i64(in_b[0]);
			bench_box::simd::store2i64(out[0], bench_box::simd::mul2i64(va, vb));
			EXPECT_EQ(static_cast<std::uint64_t>(out[0]), a * b);
			EXPECT_EQ(static_cast<std::uint64_t>(out[1]), a * b);

			bench_box::simd::store2i64(out[0], bench_box::simd::mulu2i64(va, vb));
			EXPECT_EQ(static_cast<std::uint64_t>(out[0]), a * b);
			EXPECT_EQ(static_cast<std::uint64_t>(out[1]), a * b);
		}
	}

	std::mt19937_64 rng(0x123456789abcdef0ull);
	for (int i = 0; i < 1000; i++) {
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_a[2] = {static_cast<std::int64_t>(rng()), static_cast<std::int64_t>(rng())};
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_b[2] = {static_cast<std::int64_t>(rng()), static_cast<std::int64_t>(rng())};
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t out[2];

		bench_box::simd::store2i64(out[0], bench_box::simd::mul2i64(bench_box::simd::load2i64(in_a[0]), bench_box::simd::load2i64(in_b[0])));
		for (int lane = 0; lane < 2; lane++) {
			EXPECT_EQ(static_cast<std::uint64_t>(out[lane]), static_cast<std::uint64_t>(in_a[lane]) * static_cast<std::uint64_t>(in_b[lane]));
		}
	}
}
#endif

#if defined(BENCH_BOX_SIMD_REG4I64)
TEST(SimdI64, Mul4) {
	for (const std::uint64_t a : edge_values) {
		for (const std::uint64_t b : edge_values) {
			alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_a[4] = {static_cast<std::int64_t>(a), static_cast<std::int64_t>(a), static_cast<std::int64_t>(a), static_cast<std::int64_t>(a)};
			alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_b[4] = {static_cast<std::int64_t>(b), static_cast<std::int64_t>(b), static_cast<std::int64_t>(b), static_cast<std::int64_t>(b)};
			alignas(alignof(bench_box::simd::reg4i64)) std::int64_t out[4];

			const bench_box::simd::reg4i64 va = bench_box::simd::load4i64(in_a[0]);
			const bench_box::simd::reg4i64 vb = bench_box::simd::load4i64(in_b[0]);
			bench_box::simd::store4i64(out[0], bench_box::simd::mul4i64(va, vb));
			for (const std::int64_t value : out) {
				EXPECT_EQ(static_cast<std::uint64_t>(value), a * b);
			}

			bench_box::simd::store4i64(out[0], bench_box::simd::mulu4i64(va, vb));
			for (const std::int64_t value : out) {
				EXPECT_EQ(static_cast<std::uint64_t>(value), a * b);
			}
		}
	}

	std::mt19937_64 rng(0xfedcba9876543210ull);
	for (int i = 0; i < 1000; i++) {
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_a[4];
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_b[4];
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t out[4];
		for (int lane = 0; lane < 4; lane++) {
			in_a[lane] = static_cast<std::int64_t>(rng());
			in_b[lane] = static_cast<std::int64_t>(rng());
		}

		bench_box::simd::store4i64(out[0], bench_box::simd::mul4i64(bench_box::simd::load4i64(in_a[0]), bench_box::simd::load4i64(in_b[0])));
		for (int lane = 0; lane < 4; lane++) {
			EXPECT_EQ(static_cast<std::uint64_t>(out[lane]), static_cast<std::uint64_t>(in_a[lane]) * static_cast<std::uint64_t>(in_b[lane]));
		}
	}
}
#endif

#if defined(BENCH_BOX_SIMD_REG2I64) && defined(BENCH_BOX_SIMD_REG4I64) && defined(__AVX512VL__) && defined(__AVX512DQ__)
TEST(SimdI64, Mullo2Vl) {
	std::mt19937_64 rng(0x13579bdf2468ace0ull);
	for (int i = 0; i < 1000; i++) {
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_a[2] = {static_cast<std::int64_t>(rng()), static_cast<std::int64_t>(rng())};
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t in_b[2] = {static_cast<std::int64_t>(rng()), static_cast<std::int64_t>(rng())};
		alignas(alignof(bench_box::simd::reg2i64)) std::int64_t out[2];

		bench_box::simd::store2i64(out[0], bench_box::simd::mullo2i64(bench_box::simd::load2i64(in_a[0]), bench_box::simd::load2i64(in_b[0])));
		for (int lane = 0; lane < 2; lane++) {
			EXPECT_EQ(static_cast<std::uint64_t>(out[lane]), static_cast<std::uint64_t>(in_a[lane]) * static_cast<std::uint64_t>(in_b[lane]));
		}
	}
}

TEST(SimdI64, Mullo4Vl) {
	std::mt19937_64 rng(0x2468ace013579bdfull);
	for (int i = 0; i < 1000; i++) {
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_a[4];
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t in_b[4];
		alignas(alignof(bench_box::simd::reg4i64)) std::int64_t out[4];
		for (int lane = 0; lane < 4; lane++) {
			in_a[lane] = static_cast<std::int64_t>(rng());
			in_b[lane] = static_cast<std::int64_t>(rng());
		}

		bench_box::simd::store4i64(out[0], bench_box::simd::mullo4i64(bench_box::simd::load4i64(in_a[0]), bench_box::simd::load4i64(in_b[0])));
		for (int lane = 0; lane < 4; lane++) {
			EXPECT_EQ(static_cast<std::uint64_t>(out[lane]), static_cast<std::uint64_t>(in_a[lane]) * static_cast<std::uint64_t>(in_b[lane]));
		}
	}
}
#endif

#if defined(BENCH_BOX_SIMD_REG8I64) && defined(__AVX512DQ__)
TEST(SimdI64, Mullo8) {
	for (const std::uint64_t a : edge_values) {
		for (const std::uint64_t b : edge_values) {
			alignas(alignof(bench_box::simd::reg8i64)) std::int64_t in_a[8];
			alignas(alignof(bench_box::simd::reg8i64)) std::int64_t in_b[8];
			alignas(alignof(bench_box::simd::reg8i64)) std::int64_t out[8];
			for (int lane = 0; lane < 8; lane++) {
				in_a[lane] = static_cast<std::int64_t>(a);
				in_b[lane] = static_cast<std::int64_t>(b);
			}

			bench_box::simd::store8i64(out[0], bench_box::simd::mullo8i64(bench_box::simd::load8i64(in_a[0]), bench_box::simd::load8i64(in_b[0])));
			for (const std::int64_t value : out) {
				EXPECT_EQ(static_cast<std::uint64_t>(value), a * b);
			}
		}
	}

	std::mt19937_64 rng(0x0f1e2d3c4b5a6978ull);
	for (int i = 0; i < 1000; i++) {
		alignas(alignof(bench_box::simd::reg8i64)) std::int64_t in_a[8];
		alignas(alignof(bench_box::simd::reg8i64)) std::int64_t in_b[8];
		alignas(alignof(bench_box::simd::reg8i64)) std::int64_t out[8];
		for (int lane = 0; lane < 8; lane++) {
			in_a[lane] = static_cast<std::int64_t>(rng());
			in_b[lane] = static_cast<std::int64_t>(rng());
		}

		bench_box::simd::store8i64(out[0], bench_box::simd::mullo8i64(bench_box::simd::load8i64(in_a[0]), bench_box::simd::load8i64(in_b[0])));
		for (int lane = 0; lane < 8; lane++) {
			EXPECT_EQ(static_cast<std::uint64_t>(out[lane]), static_cast<std::uint64_t>(in_a[lane]) * static_cast<std::uint64_t>(in_b[lane]));
		}
	}
}
#endif
