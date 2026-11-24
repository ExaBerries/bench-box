#include <gtest/gtest.h>
#include <bench_box/math/vec2.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

template <typename V>
concept square_root_api = requires(V v) {
	{ v.mag() };
	{ v.nor() };
	{ v.dst(v) };
};

static_assert(square_root_api<bench_box::math::vec2f>);
static_assert(square_root_api<bench_box::math::vec2d>);
static_assert(!square_root_api<bench_box::math::vec2i>);
static_assert(!square_root_api<bench_box::math::vec2li>);

TEST(Vec2F, Layout) {
	using vec2f = bench_box::math::vec2f;
#if defined(BENCH_BOX_SIMD_REG4F)
	static_assert(sizeof(vec2f) == 16);
	static_assert(alignof(vec2f) == 16);
#else
	static_assert(sizeof(vec2f) == 8);
	static_assert(alignof(vec2f) == alignof(float));
#endif
}

TEST(Vec2F, Arithmetic) {
	using vec2f = bench_box::math::vec2f;

	static_assert(vec2f{1.0f, 2.0f} + vec2f{4.0f, 5.0f} == vec2f{5.0f, 7.0f});
	static_assert(vec2f{4.0f, 5.0f} - vec2f{1.0f, 2.0f} == vec2f{3.0f, 3.0f});
	static_assert(vec2f{1.0f, 2.0f} * 2.0f == vec2f{2.0f, 4.0f});
	static_assert(vec2f{1.0f, 2.0f} / 2.0f == vec2f{0.5f, 1.0f});
	static_assert(vec2f{1.0f, 2.0f} + 1.0f == vec2f{2.0f, 3.0f});
	static_assert(vec2f{1.0f, 2.0f} - 1.0f == vec2f{0.0f, 1.0f});
	static_assert(vec2f{2.0f}.mag2() == 8.0f);

	vec2f a{1.0f, 2.0f};
	a += vec2f{0.5f, 0.5f};
	EXPECT_TRUE((a == vec2f{1.5f, 2.5f}));
	a -= vec2f{0.5f, 0.5f};
	a *= 2.0f;
	EXPECT_TRUE((a == vec2f{2.0f, 4.0f}));
	a /= 2.0f;
	EXPECT_TRUE((a == vec2f{1.0f, 2.0f}));
	a += 1.0f;
	EXPECT_TRUE((a == vec2f{2.0f, 3.0f}));
	a -= 1.0f;
	EXPECT_TRUE((a == vec2f{1.0f, 2.0f}));
	EXPECT_TRUE((a != vec2f{1.0f, 4.0f}));

	vec2f s{};
	s.set(3.25f, -0.5f);
	EXPECT_TRUE((s == vec2f{3.25f, -0.5f}));
	s.set(vec2f{1.25f, 2.0f});
	EXPECT_TRUE((s == vec2f{1.25f, 2.0f}));
}

TEST(Vec2F, Geometry) {
	using vec2f = bench_box::math::vec2f;

	static_assert(vec2f{1.0f, 2.0f}.mag2() == 5.0f);
	static_assert(vec2f{1.0f, 2.0f}.dot(vec2f{4.0f, 5.0f}) == 14.0f);
	static_assert(vec2f{1.0f, 2.0f}.dst2(vec2f{1.0f, 2.0f}) == 0.0f);

	vec2f p{3.0f, 4.0f};
	EXPECT_FLOAT_EQ(p.mag(), 5.0f);
	vec2f o{};
	vec2f q{0.0f, 4.0f};
	vec2f r{3.0f, 0.0f};
	EXPECT_FLOAT_EQ(q.dst(r), 5.0f);
	EXPECT_FLOAT_EQ(o.dst2(p), 25.0f);
	EXPECT_FLOAT_EQ(p.dot(vec2f{2.0f, 0.0f}), 6.0f);

	static_assert(vec2f{1.0f, 0.0f}.crs(vec2f{0.0f, 1.0f}) == 1.0f);
	static_assert(vec2f{0.0f, 1.0f}.crs(vec2f{1.0f, 0.0f}) == -1.0f);
	static_assert(vec2f{2.0f, 1.0f}.crs(vec2f{4.0f, 2.0f}) == 0.0f);

	vec2f n{2.0f, 0.0f};
	n.nor();
	EXPECT_TRUE((n == vec2f{1.0f, 0.0f}));
	EXPECT_FLOAT_EQ(n.mag(), 1.0f);

	vec2f z{};
	z.nor();
	EXPECT_TRUE((z == vec2f{0.0f, 0.0f}));
	vec2f u{0.0f, 1.0f};
	u.nor();
	EXPECT_TRUE((u == vec2f{0.0f, 1.0f}));
}

TEST(Vec2D, Layout) {
	using vec2d = bench_box::math::vec2d;
#if defined(BENCH_BOX_SIMD_REG2D)
	static_assert(sizeof(vec2d) == 16);
	static_assert(alignof(vec2d) == 16);
#else
	static_assert(sizeof(vec2d) == 16);
	static_assert(alignof(vec2d) == alignof(double));
#endif
}

TEST(Vec2D, Arithmetic) {
	using vec2d = bench_box::math::vec2d;

	static_assert(vec2d{1.0, 2.0} + vec2d{4.0, 5.0} == vec2d{5.0, 7.0});
	static_assert(vec2d{4.0, 5.0} - vec2d{1.0, 2.0} == vec2d{3.0, 3.0});
	static_assert(vec2d{1.0, 2.0} * 2.0 == vec2d{2.0, 4.0});
	static_assert(vec2d{1.0, 2.0} / 2.0 == vec2d{0.5, 1.0});
	static_assert(vec2d{1.0, 2.0} + 1.0 == vec2d{2.0, 3.0});
	static_assert(vec2d{1.0, 2.0} - 1.0 == vec2d{0.0, 1.0});
	static_assert(vec2d{2.0}.mag2() == 8.0);

	vec2d a{1.0, 2.0};
	a += vec2d{0.5, 0.5};
	EXPECT_TRUE((a == vec2d{1.5, 2.5}));
	a -= vec2d{0.5, 0.5};
	a *= 2.0;
	EXPECT_TRUE((a == vec2d{2.0, 4.0}));
	a /= 2.0;
	EXPECT_TRUE((a == vec2d{1.0, 2.0}));
	a += 1.0;
	EXPECT_TRUE((a == vec2d{2.0, 3.0}));
	a -= 1.0;
	EXPECT_TRUE((a == vec2d{1.0, 2.0}));
	EXPECT_TRUE((a != vec2d{1.0, 4.0}));

	vec2d s{};
	s.set(3.25, -0.5);
	EXPECT_TRUE((s == vec2d{3.25, -0.5}));
	s.set(vec2d{1.25, 2.0});
	EXPECT_TRUE((s == vec2d{1.25, 2.0}));
}

TEST(Vec2D, Geometry) {
	using vec2d = bench_box::math::vec2d;

	static_assert(vec2d{1.0, 2.0}.mag2() == 5.0);
	static_assert(vec2d{1.0, 2.0}.dot(vec2d{4.0, 5.0}) == 14.0);
	static_assert(vec2d{1.0, 2.0}.dst2(vec2d{1.0, 2.0}) == 0.0);

	vec2d p{3.0, 4.0};
	EXPECT_DOUBLE_EQ(p.mag(), 5.0);
	vec2d o{};
	vec2d q{0.0, 4.0};
	vec2d r{3.0, 0.0};
	EXPECT_DOUBLE_EQ(q.dst(r), 5.0);
	EXPECT_DOUBLE_EQ(o.dst2(p), 25.0);
	EXPECT_DOUBLE_EQ(p.dot(vec2d{2.0, 0.0}), 6.0);

	static_assert(vec2d{1.0, 0.0}.crs(vec2d{0.0, 1.0}) == 1.0);
	static_assert(vec2d{0.0, 1.0}.crs(vec2d{1.0, 0.0}) == -1.0);
	static_assert(vec2d{2.0, 1.0}.crs(vec2d{4.0, 2.0}) == 0.0);

	vec2d n{2.0, 0.0};
	n.nor();
	EXPECT_TRUE((n == vec2d{1.0, 0.0}));
	EXPECT_DOUBLE_EQ(n.mag(), 1.0);

	vec2d z{};
	z.nor();
	EXPECT_TRUE((z == vec2d{0.0, 0.0}));
	vec2d u{0.0, 1.0};
	u.nor();
	EXPECT_TRUE((u == vec2d{0.0, 1.0}));
}

TEST(Vec2I, Layout) {
	using vec2i = bench_box::math::vec2i;
#if defined(BENCH_BOX_SIMD_REG4I32)
	static_assert(sizeof(vec2i) == 16);
	static_assert(alignof(vec2i) == 16);
#else
	static_assert(sizeof(vec2i) == 8);
	static_assert(alignof(vec2i) == alignof(std::int32_t));
#endif
}

TEST(Vec2I, Arithmetic) {
	using vec2i = bench_box::math::vec2i;

	static_assert(vec2i{1, 2} + vec2i{4, 5} == vec2i{5, 7});
	static_assert(vec2i{4, 5} - vec2i{1, 2} == vec2i{3, 3});
	static_assert(vec2i{1, 2} * 2 == vec2i{2, 4});
	static_assert(vec2i{1, 2} / 2 == vec2i{0, 1});
	static_assert(vec2i{1, 2} + 1 == vec2i{2, 3});
	static_assert(vec2i{1, 2} - 1 == vec2i{0, 1});
	static_assert(vec2i{2}.mag2() == 8);

	vec2i a{1, 2};
	a += vec2i{1, 1};
	EXPECT_TRUE((a == vec2i{2, 3}));
	a -= vec2i{1, 1};
	a *= 3;
	EXPECT_TRUE((a == vec2i{3, 6}));
	a /= 3;
	EXPECT_TRUE((a == vec2i{1, 2}));
	a += 1;
	EXPECT_TRUE((a == vec2i{2, 3}));
	a -= 1;
	EXPECT_TRUE((a == vec2i{1, 2}));
	EXPECT_TRUE((a != vec2i{1, 4}));

	vec2i s{};
	s.set(7, -8);
	EXPECT_TRUE((s == vec2i{7, -8}));
	s.set(vec2i{1, 2});
	EXPECT_TRUE((s == vec2i{1, 2}));
}

TEST(Vec2I, Geometry) {
	using vec2i = bench_box::math::vec2i;

	static_assert(vec2i{1, 2}.mag2() == 5);
	static_assert(vec2i{-7, -8}.mag2() == 113);
	static_assert(vec2i{1, 2}.dot(vec2i{4, 5}) == 14);
	static_assert(vec2i{1, 2}.dst2(vec2i{2, 2}) == 1);
	static_assert(vec2i{3, 4}.crs(vec2i{5, 6}) == -2);

	vec2i p{0, 4};
	EXPECT_EQ(p.mag2(), 16);
	vec2i o{};
	EXPECT_EQ(o.dst2(p), 16);

	vec2i x{1000, 2000};
	EXPECT_EQ(x.mag2(), 5000000);
}

TEST(Vec2LI, Layout) {
	using vec2li = bench_box::math::vec2li;
#if defined(BENCH_BOX_SIMD_REG2I64)
	static_assert(sizeof(vec2li) == 16);
	static_assert(alignof(vec2li) == 16);
#else
	static_assert(sizeof(vec2li) == 16);
	static_assert(alignof(vec2li) == alignof(std::int64_t));
#endif
}

TEST(Vec2LI, Arithmetic) {
	using vec2li = bench_box::math::vec2li;

	constexpr int64_t two = 2;
	constexpr int64_t three = 3;

	static_assert(vec2li{1, 2} + vec2li{4, 5} == vec2li{5, 7});
	static_assert(vec2li{1, 2} * two == vec2li{2, 4});
	static_assert(vec2li{1, 2} / two == vec2li{0, 1});
	static_assert(vec2li{1, 2} + three == vec2li{4, 5});
	static_assert(vec2li{1, 2} - three == vec2li{-2, -1});
	static_assert(vec2li{2}.mag2() == 8);

	vec2li a{1, 2};
	a += vec2li{1, 1};
	EXPECT_TRUE((a == vec2li{2, 3}));
	a -= vec2li{1, 1};
	a *= 3;
	EXPECT_TRUE((a == vec2li{3, 6}));
	a /= 3;
	EXPECT_TRUE((a == vec2li{1, 2}));
	EXPECT_TRUE((a != vec2li{1, 4}));

	vec2li s{};
	s.set(7, -8);
	EXPECT_TRUE((s == vec2li{7, -8}));
	s.set(vec2li{1, 2});
	EXPECT_TRUE((s == vec2li{1, 2}));
}

TEST(Vec2LI, Geometry) {
	using vec2li = bench_box::math::vec2li;

	static_assert(vec2li{100000, 200000}.mag2() == 50000000000);
	static_assert(vec2li{100000, 200000}.dot(vec2li{1, 2}) == 500000);
	static_assert(vec2li{0, 0}.dst2(vec2li{100000, 200000}) == 50000000000);
	static_assert(vec2li{100000, 200000}.crs(vec2li{300000, 400000}) == -20000000000);

	vec2li p{100000, 200000};
	EXPECT_EQ(p.mag2(), 50000000000);
	EXPECT_EQ(p.crs(vec2li{300000, 400000}), -20000000000);
}

TEST(Vec2F, MinMaxClamp) {
	using bench_box::math::clamp;
	using bench_box::math::max;
	using bench_box::math::min;
	using vec2f = bench_box::math::vec2f;

	static_assert(min(vec2f{1.0f, 4.0f}, vec2f{3.0f, 2.0f}) == vec2f{1.0f, 2.0f});
	static_assert(max(vec2f{1.0f, 4.0f}, vec2f{3.0f, 2.0f}) == vec2f{3.0f, 4.0f});
	static_assert(clamp(vec2f{-1.0f, 5.0f}, vec2f{0.0f, 0.0f}, vec2f{3.0f, 3.0f}) == vec2f{0.0f, 3.0f});
	static_assert(clamp(vec2f{1.0f, 2.0f}, vec2f{0.0f, 0.0f}, vec2f{3.0f, 3.0f}) == vec2f{1.0f, 2.0f});
	static_assert(clamp(vec2f{-1.0f, 5.0f}, 0.0f, 3.0f) == vec2f{0.0f, 3.0f});
}

TEST(Vec2I, MinMaxClamp) {
	using bench_box::math::clamp;
	using bench_box::math::max;
	using bench_box::math::min;
	using vec2i = bench_box::math::vec2i;

	static_assert(min(vec2i{1, 4}, vec2i{3, 2}) == vec2i{1, 2});
	static_assert(max(vec2i{1, 4}, vec2i{3, 2}) == vec2i{3, 4});
	static_assert(clamp(vec2i{-1, 5}, vec2i{0, 0}, vec2i{3, 3}) == vec2i{0, 3});
	static_assert(clamp(vec2i{1, 2}, vec2i{0, 0}, vec2i{3, 3}) == vec2i{1, 2});
	static_assert(clamp(vec2i{-1, 5}, 0, 3) == vec2i{0, 3});
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
