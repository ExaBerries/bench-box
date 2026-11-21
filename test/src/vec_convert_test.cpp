#include <gtest/gtest.h>
#include <bench_box/math/math.h>
#include <bench_box/math/vec_convert.h>
#include <cmath>
#include <cstdint>
#include <limits>

using vec2d = bench_box::math::vec2d;
using vec2f = bench_box::math::vec2f;
using vec2i = bench_box::math::vec2i;
using vec2li = bench_box::math::vec2li;
using vec3d = bench_box::math::vec3d;
using vec3f = bench_box::math::vec3f;
using vec3i = bench_box::math::vec3i;
using vec3li = bench_box::math::vec3li;

using bench_box::math::checked_vector_cast;
using bench_box::math::round_vector_cast;
using bench_box::math::vector_cast;

TEST(VecConvert, Widening) {
	static_assert(vector_cast<vec3d>(vec3i{1, -2, 3}) == vec3d{1.0, -2.0, 3.0});
	static_assert(vector_cast<vec3d>(vec3i{2147483647, -2147483648, 0}) == vec3d{2147483647.0, -2147483648.0, 0.0});
	static_assert(vector_cast<vec3d>(vec3f{0.5f, -0.25f, 8.0f}) == vec3d{0.5, -0.25, 8.0});

	static_assert(vector_cast<vec3f>(vector_cast<vec3d>(vec3f{0.1f, 0.2f, 0.3f})) == vec3f{0.1f, 0.2f, 0.3f});

	static_assert(vector_cast<vec3f>(vec3i{1, 2, 3}) == vec3f{1.0f, 2.0f, 3.0f});
	static_assert(vector_cast<vec3li>(vec3i{2147483647, -2147483648, 0}) == vec3li{2147483647, -2147483648, 0});
}

TEST(VecConvert, Narrowing) {
	static_assert(vector_cast<vec3f>(vec3d{0.5, -0.25, 8.0}) == vec3f{0.5f, -0.25f, 8.0f});

	static_assert(vector_cast<vec3f>(vec3d{16777217.0, 0.0, 0.0}) == vec3f{16777216.0f, 0.0f, 0.0f});
	static_assert(vector_cast<vec3f>(vec3i{16777217, 0, 0}) == vec3f{16777216.0f, 0.0f, 0.0f});
	static_assert(vector_cast<vec3f>(vec3i{2147483647, 0, 0}) == vec3f{2147483648.0f, 0.0f, 0.0f});
	EXPECT_FALSE(checked_vector_cast<vec3i>(vector_cast<vec3f>(vec3i{2147483647, 0, 0})).has_value());

	static_assert(vector_cast<vec3f>(vec3d{1e300, 0.0, 0.0}) == vec3f{std::numeric_limits<float>::infinity(), 0.0f, 0.0f});
	static_assert(vector_cast<vec3f>(vec3d{std::numeric_limits<double>::denorm_min(), 0.0, 0.0}) == vec3f{0.0f, 0.0f, 0.0f});
}

TEST(VecConvert, Truncate) {
	static_assert(vector_cast<vec3i>(vec3f{1.9f, -1.9f, 0.5f}) == vec3i{1, -1, 0});
	static_assert(vector_cast<vec3i>(vec3f{-0.5f, 2.7f, -2.7f}) == vec3i{0, 2, -2});
	static_assert(vector_cast<vec3i>(vec3f{3.0f, -3.0f, 0.0f}) == vec3i{3, -3, 0});
	static_assert(vector_cast<vec3i>(vec3d{1.9, -1.9, 0.5}) == vec3i{1, -1, 0});
	static_assert(vector_cast<vec3i>(vec3d{-0.5, 2.7, -2.7}) == vec3i{0, 2, -2});
	static_assert(vector_cast<vec3i>(vec3d{2147483647.0, -2147483648.0, 0.0}) == vec3i{2147483647, -2147483648, 0});

	static_assert(vector_cast<vec3i>(vec3li{2147483648, 0, 0}) == vec3i{-2147483648, 0, 0});

	static_assert(vector_cast<vec3i>(vector_cast<vec3f>(vec3i{1, -2, 3})) == vec3i{1, -2, 3});
}

TEST(VecConvert, Round) {
	EXPECT_TRUE((round_vector_cast<vec3i>(vec3f{0.5f, 1.5f, 2.5f}) == vec3i{1, 2, 3}));
	EXPECT_TRUE((round_vector_cast<vec3i>(vec3f{-0.5f, -1.5f, 2.4f}) == vec3i{-1, -2, 2}));
	EXPECT_TRUE((round_vector_cast<vec3i>(vec3d{-2.5, 0.4999, 0.5001}) == vec3i{-3, 0, 1}));
	EXPECT_TRUE((round_vector_cast<vec3i>(vec3d{1.9, -1.9, 0.0}) == vec3i{2, -2, 0}));

	EXPECT_TRUE((vector_cast<vec3i>(vec3f{-0.5f, 0.5f, 1.5f}) == vec3i{0, 0, 1}));
	EXPECT_TRUE((round_vector_cast<vec3i>(vec3f{-0.5f, 0.5f, 1.5f}) == vec3i{-1, 1, 2}));

	EXPECT_TRUE((round_vector_cast<vec3li>(vec3d{0.5, -0.5, 3000000000.5}) == vec3li{1, -1, 3000000001}));
}

TEST(VecConvert, Checked) {
	const auto truncated_f = checked_vector_cast<vec3i>(vec3f{1.9f, -1.9f, 0.5f});
	ASSERT_TRUE(truncated_f.has_value());
	EXPECT_TRUE((truncated_f.value() == vec3i{1, -1, 0}));

	const auto truncated_d = checked_vector_cast<vec3i>(vec3d{1.9, -1.9, 0.5});
	ASSERT_TRUE(truncated_d.has_value());
	EXPECT_TRUE((truncated_d.value() == vec3i{1, -1, 0}));

	const auto bounds = checked_vector_cast<vec3i>(vec3d{2147483647.0, -2147483648.0, 0.0});
	ASSERT_TRUE(bounds.has_value());
	EXPECT_TRUE((bounds.value() == vec3i{2147483647, -2147483648, 0}));

	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double inf = std::numeric_limits<double>::infinity();
	const float nan_f = std::numeric_limits<float>::quiet_NaN();
	const float inf_f = std::numeric_limits<float>::infinity();

	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{2147483648.0, 0.0, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{-2147483649.0, 0.0, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{0.0, 1e300, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{0.0, 0.0, nan}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{inf, 0.0, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3d{0.0, 0.0, -inf}).has_value());

	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{2147483648.0f, 0.0f, 0.0f}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{-4294967296.0f, 0.0f, 0.0f}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{0.0f, 0.0f, nan_f}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{0.0f, inf_f, 0.0f}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{0.0f, 0.0f, -inf_f}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3i>(vec3f{std::numeric_limits<float>::max(), 0.0f, 0.0f}).has_value());

	const auto round_trip = checked_vector_cast<vec3i>(vector_cast<vec3d>(vec3i{1, -2, 3}));
	ASSERT_TRUE(round_trip.has_value());
	EXPECT_TRUE((round_trip.value() == vec3i{1, -2, 3}));
}

TEST(VecConvert, Checked64) {
	constexpr double i64_lowest = -9223372036854775808.0;
	constexpr double i64_past = 9223372036854775808.0;

	const auto lowest = checked_vector_cast<vec3li>(vec3d{i64_lowest, 0.0, 1.5});
	ASSERT_TRUE(lowest.has_value());
	EXPECT_EQ(lowest.value().x, std::numeric_limits<int64_t>::lowest());
	EXPECT_EQ(lowest.value().z, 1);

	EXPECT_FALSE(checked_vector_cast<vec3li>(vec3d{i64_past, 0.0, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3li>(vec3d{i64_lowest - 2048.0, 0.0, 0.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec3li>(vec3d{0.0, 1e300, 0.0}).has_value());

	const auto wide = vec3d{4000000000.0, 0.0, 0.0};
	EXPECT_FALSE(checked_vector_cast<vec3i>(wide).has_value());
	const auto wide64 = checked_vector_cast<vec3li>(wide);
	ASSERT_TRUE(wide64.has_value());
	EXPECT_EQ(wide64.value().x, 4000000000);
}

TEST(VecConvert, TwoElements) {
	static_assert(vector_cast<vec2d>(vec2i{1, -2}) == vec2d{1.0, -2.0});
	static_assert(vector_cast<vec2i>(vec2f{1.9f, -1.9f}) == vec2i{1, -1});
	static_assert(vector_cast<vec2i>(vec2d{1.9, -1.9}) == vec2i{1, -1});
	static_assert(vector_cast<vec2f>(vector_cast<vec2d>(vec2f{0.1f, 0.2f})) == vec2f{0.1f, 0.2f});
	static_assert(vector_cast<vec2li>(vec2i{2147483647, -2147483648}) == vec2li{2147483647, -2147483648});

	EXPECT_TRUE((round_vector_cast<vec2i>(vec2d{-2.5, 0.5001}) == vec2i{-3, 1}));
	EXPECT_TRUE((round_vector_cast<vec2li>(vec2f{0.5f, -0.5f}) == vec2li{1, -1}));

	const auto bounds = checked_vector_cast<vec2i>(vec2d{2147483647.0, -2147483648.0});
	ASSERT_TRUE(bounds.has_value());
	EXPECT_TRUE((bounds.value() == vec2i{2147483647, -2147483648}));

	EXPECT_FALSE(checked_vector_cast<vec2i>(vec2d{0.0, 2147483648.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec2li>(vec2d{0.0, 9223372036854775808.0}).has_value());
	EXPECT_FALSE(checked_vector_cast<vec2i>(vec2f{0.0f, std::numeric_limits<float>::infinity()}).has_value());
}
