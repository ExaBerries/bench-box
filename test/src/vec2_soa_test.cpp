#include <gtest/gtest.h>
#include <bench_box/memory/vec2_soa.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

using bench_box::vec2d_soa;
using bench_box::vec2f_soa;
using bench_box::vec2i_soa;
using bench_box::vec2li_soa;

static_assert(std::is_same_v<typename vec2f_soa::vector_t, bench_box::math::vec2f>);
static_assert(std::is_same_v<typename vec2d_soa::vector_t, bench_box::math::vec2d>);
static_assert(std::is_same_v<typename vec2i_soa::vector_t, bench_box::math::vec2i>);
static_assert(std::is_same_v<typename vec2li_soa::vector_t, bench_box::math::vec2li>);
static_assert(!std::is_copy_constructible_v<vec2f_soa>);
static_assert(!std::is_copy_assignable_v<vec2f_soa>);
static_assert(std::is_nothrow_move_constructible_v<vec2f_soa>);
static_assert(std::is_nothrow_move_assignable_v<vec2f_soa>);

TEST(Vec2Soa, EmptyAndSized) {
	vec2f_soa empty;
	EXPECT_TRUE(empty.empty());
	EXPECT_EQ(empty.size(), 0u);
	EXPECT_FALSE(empty.operator bool());
	EXPECT_EQ(empty.x, nullptr);
	EXPECT_EQ(empty.y, nullptr);
	EXPECT_TRUE(empty.x_span().empty());
	EXPECT_TRUE(empty.y_span().empty());

	vec2f_soa values(4);
	ASSERT_TRUE(values.valid());
	EXPECT_FALSE(values.empty());
	EXPECT_TRUE(values.operator bool());
	EXPECT_EQ(values.size(), 4u);
	EXPECT_EQ(values.x_span().size(), 4u);
	EXPECT_EQ(values.y_span().size(), 4u);
	EXPECT_EQ(values.x + values.size(), values.y);
}

TEST(Vec2Soa, ComponentAccessAndSetGet) {
	vec2f_soa values(3);

	values.x_span()[0] = 1.0f;
	values.x_span()[1] = 2.0f;
	values.x_span()[2] = 3.0f;
	values.y_span()[0] = 4.0f;
	values.y_span()[1] = 5.0f;
	values.y_span()[2] = 6.0f;

	EXPECT_TRUE((values.get(0) == bench_box::math::vec2f{1.0f, 4.0f}));
	EXPECT_TRUE((values.get(1) == bench_box::math::vec2f{2.0f, 5.0f}));
	EXPECT_TRUE((values.get(2) == bench_box::math::vec2f{3.0f, 6.0f}));

	values.set(1, bench_box::math::vec2f{8.0f, 9.0f});
	EXPECT_TRUE((values.get(1) == bench_box::math::vec2f{8.0f, 9.0f}));

	values.set(2, 10.0f, 11.0f);
	EXPECT_TRUE((values.get(2) == bench_box::math::vec2f{10.0f, 11.0f}));

	const vec2f_soa& const_values = values;
	EXPECT_TRUE((const_values[0] == bench_box::math::vec2f{1.0f, 4.0f}));
}

TEST(Vec2Soa, Reference) {
	vec2f_soa values(2);

	values[0] = bench_box::math::vec2f{1.0f, 2.0f};
	values[1] = bench_box::math::vec2f{3.0f, 4.0f};
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{1.0f, 2.0f}));

	values[0] += bench_box::math::vec2f{1.0f, 1.0f};
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{2.0f, 3.0f}));

	values[0] -= 1.0f;
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{1.0f, 2.0f}));

	values[0] *= 3.0f;
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{3.0f, 6.0f}));

	values[0] /= 3.0f;
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{1.0f, 2.0f}));

	values[0].x = 7.0f;
	values[0].y = 8.0f;
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{7.0f, 8.0f}));

	values[0] = values[1];
	EXPECT_TRUE((bench_box::math::vec2f(values[0]) == bench_box::math::vec2f{3.0f, 4.0f}));
}

TEST(Vec2Soa, ConstructionAndFill) {
	const bench_box::math::vec2f source[3] = {
		{1.0f, 2.0f},
		{3.0f, 4.0f},
		{5.0f, 6.0f}
	};

	vec2f_soa values(std::span<const bench_box::math::vec2f>(source, 3));
	ASSERT_EQ(values.size(), 3u);
	EXPECT_TRUE((values.get(0) == bench_box::math::vec2f{1.0f, 2.0f}));
	EXPECT_TRUE((values.get(2) == bench_box::math::vec2f{5.0f, 6.0f}));

	values.fill(bench_box::math::vec2f{9.0f, 10.0f});
	EXPECT_TRUE((values.get(1) == bench_box::math::vec2f{9.0f, 10.0f}));

	values.fill(11.0f, 12.0f);
	EXPECT_TRUE((values.get(1) == bench_box::math::vec2f{11.0f, 12.0f}));
}

TEST(Vec2Soa, IntegralTypes) {
	vec2i_soa integers(2);
	integers.fill(1, 2);
	integers[1] = bench_box::math::vec2i{3, 4};
	integers[0] *= 2;
	integers[1] *= 2;

	EXPECT_TRUE((integers.get(0) == bench_box::math::vec2i{2, 4}));
	EXPECT_TRUE((integers.get(1) == bench_box::math::vec2i{6, 8}));

	vec2li_soa wide(1);
	wide.set(0, std::int64_t{7}, std::int64_t{8});
	EXPECT_TRUE((wide.get(0) == bench_box::math::vec2li{7, 8}));
}

TEST(Vec2Soa, Move) {
	vec2f_soa first(2);
	first.fill(1.0f, 2.0f);

	vec2f_soa second(std::move(first));
	EXPECT_TRUE(first.empty());
	EXPECT_EQ(first.x, nullptr);
	EXPECT_EQ(first.y, nullptr);
	EXPECT_NE(second.x, nullptr);
	EXPECT_EQ(second.x + second.size(), second.y);
	EXPECT_TRUE((second.get(0) == bench_box::math::vec2f{1.0f, 2.0f}));

	vec2f_soa third;
	third = std::move(second);
	EXPECT_TRUE(second.empty());
	EXPECT_EQ(second.x, nullptr);
	EXPECT_EQ(second.y, nullptr);
	EXPECT_NE(third.x, nullptr);
	EXPECT_EQ(third.x + third.size(), third.y);
	EXPECT_TRUE((third.get(1) == bench_box::math::vec2f{1.0f, 2.0f}));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
