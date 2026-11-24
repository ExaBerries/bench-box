#include <gtest/gtest.h>
#include <bench_box/math/vec3.h>

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

static_assert(square_root_api<bench_box::math::vec3f>);
static_assert(square_root_api<bench_box::math::vec3d>);
static_assert(!square_root_api<bench_box::math::vec3i>);
static_assert(!square_root_api<bench_box::math::vec3li>);

TEST(Vec3F, Layout) {
	using vec3f = bench_box::math::vec3f;
#if defined(BENCH_BOX_SIMD_REG4F)
	static_assert(sizeof(vec3f) == 16);
	static_assert(alignof(vec3f) == 16);
#else
	static_assert(sizeof(vec3f) == 12);
	static_assert(alignof(vec3f) == alignof(float));
#endif
}

TEST(Vec3F, Arithmetic) {
	using vec3f = bench_box::math::vec3f;

	static_assert(vec3f{1.0f, 2.0f, 3.0f} + vec3f{4.0f, 5.0f, 6.0f} == vec3f{5.0f, 7.0f, 9.0f});
	static_assert(vec3f{4.0f, 5.0f, 6.0f} - vec3f{1.0f, 2.0f, 3.0f} == vec3f{3.0f, 3.0f, 3.0f});
	static_assert(vec3f{1.0f, 2.0f, 3.0f} * 2.0f == vec3f{2.0f, 4.0f, 6.0f});
	static_assert(vec3f{1.0f, 2.0f, 3.0f} / 2.0f == vec3f{0.5f, 1.0f, 1.5f});
	static_assert(vec3f{1.0f, 2.0f, 3.0f} + 1.0f == vec3f{2.0f, 3.0f, 4.0f});
	static_assert(vec3f{1.0f, 2.0f, 3.0f} - 1.0f == vec3f{0.0f, 1.0f, 2.0f});
	static_assert(vec3f{2.0f}.mag2() == 12.0f);

	vec3f a{1.0f, 2.0f, 3.0f};
	a += vec3f{0.5f, 0.5f, 0.5f};
	EXPECT_TRUE((a == vec3f{1.5f, 2.5f, 3.5f}));
	a -= vec3f{0.5f, 0.5f, 0.5f};
	a *= 2.0f;
	EXPECT_TRUE((a == vec3f{2.0f, 4.0f, 6.0f}));
	a /= 2.0f;
	EXPECT_TRUE((a == vec3f{1.0f, 2.0f, 3.0f}));
	a += 1.0f;
	EXPECT_TRUE((a == vec3f{2.0f, 3.0f, 4.0f}));
	a -= 1.0f;
	EXPECT_TRUE((a == vec3f{1.0f, 2.0f, 3.0f}));
	EXPECT_TRUE((a != vec3f{1.0f, 2.0f, 4.0f}));

	vec3f s{};
	s.set(3.25f, -0.5f, 8.0f);
	EXPECT_TRUE((s == vec3f{3.25f, -0.5f, 8.0f}));
	s.set(vec3f{1.25f, 2.0f, 3.0f});
	EXPECT_TRUE((s == vec3f{1.25f, 2.0f, 3.0f}));
}

TEST(Vec3F, Geometry) {
	using vec3f = bench_box::math::vec3f;

	static_assert(vec3f{1.0f, 2.0f, 3.0f}.mag2() == 14.0f);
	static_assert(vec3f{1.0f, 2.0f, 3.0f}.dot(vec3f{4.0f, 5.0f, 6.0f}) == 32.0f);
	static_assert(vec3f{1.0f, 2.0f, 3.0f}.dst2(vec3f{1.0f, 2.0f, 3.0f}) == 0.0f);

	vec3f p{0.0f, 3.0f, 0.0f};
	EXPECT_FLOAT_EQ(p.mag(), 3.0f);
	vec3f o{};
	vec3f q{0.0f, 4.0f, 3.0f};
	EXPECT_FLOAT_EQ(o.dst(q), 5.0f);
	EXPECT_FLOAT_EQ(q.dot(p), 12.0f);

	vec3f c{1.0f, 0.0f, 0.0f};
	c.crs(vec3f{0.0f, 1.0f, 0.0f});
	EXPECT_TRUE((c == vec3f{0.0f, 0.0f, 1.0f}));

	vec3f n{2.0f, 0.0f, 0.0f};
	n.nor();
	EXPECT_TRUE((n == vec3f{1.0f, 0.0f, 0.0f}));
	EXPECT_FLOAT_EQ(n.mag(), 1.0f);

	vec3f z{};
	z.nor();
	EXPECT_TRUE((z == vec3f{0.0f, 0.0f, 0.0f}));
	vec3f u{0.0f, 1.0f, 0.0f};
	u.nor();
	EXPECT_TRUE((u == vec3f{0.0f, 1.0f, 0.0f}));
}

TEST(Vec3D, Layout) {
	using vec3d = bench_box::math::vec3d;
#if defined(BENCH_BOX_SIMD_REG4D)
	static_assert(sizeof(vec3d) == 32);
	static_assert(alignof(vec3d) == 32);
#else
	static_assert(sizeof(vec3d) == 24);
	static_assert(alignof(vec3d) == alignof(double));
#endif
}

TEST(Vec3D, Arithmetic) {
	using vec3d = bench_box::math::vec3d;

	static_assert(vec3d{1.0, 2.0, 3.0} + vec3d{4.0, 5.0, 6.0} == vec3d{5.0, 7.0, 9.0});
	static_assert(vec3d{4.0, 5.0, 6.0} - vec3d{1.0, 2.0, 3.0} == vec3d{3.0, 3.0, 3.0});
	static_assert(vec3d{1.0, 2.0, 3.0} * 2.0 == vec3d{2.0, 4.0, 6.0});
	static_assert(vec3d{1.0, 2.0, 3.0} / 2.0 == vec3d{0.5, 1.0, 1.5});
	static_assert(vec3d{1.0, 2.0, 3.0} + 1.0 == vec3d{2.0, 3.0, 4.0});
	static_assert(vec3d{1.0, 2.0, 3.0} - 1.0 == vec3d{0.0, 1.0, 2.0});
	static_assert(vec3d{2.0}.mag2() == 12.0);

	vec3d a{1.0, 2.0, 3.0};
	a += vec3d{0.5, 0.5, 0.5};
	EXPECT_TRUE((a == vec3d{1.5, 2.5, 3.5}));
	a -= vec3d{0.5, 0.5, 0.5};
	a *= 2.0;
	EXPECT_TRUE((a == vec3d{2.0, 4.0, 6.0}));
	a /= 2.0;
	EXPECT_TRUE((a == vec3d{1.0, 2.0, 3.0}));
	a += 1.0;
	EXPECT_TRUE((a == vec3d{2.0, 3.0, 4.0}));
	a -= 1.0;
	EXPECT_TRUE((a == vec3d{1.0, 2.0, 3.0}));
	EXPECT_TRUE((a != vec3d{1.0, 2.0, 4.0}));

	vec3d s{};
	s.set(3.25, -0.5, 8.0);
	EXPECT_TRUE((s == vec3d{3.25, -0.5, 8.0}));
	s.set(vec3d{1.25, 2.0, 3.0});
	EXPECT_TRUE((s == vec3d{1.25, 2.0, 3.0}));
}

TEST(Vec3D, Geometry) {
	using vec3d = bench_box::math::vec3d;

	static_assert(vec3d{1.0, 2.0, 3.0}.mag2() == 14.0);
	static_assert(vec3d{1.0, 2.0, 3.0}.dot(vec3d{4.0, 5.0, 6.0}) == 32.0);
	static_assert(vec3d{1.0, 2.0, 3.0}.dst2(vec3d{1.0, 2.0, 3.0}) == 0.0);

	vec3d p{0.0, 3.0, 0.0};
	EXPECT_DOUBLE_EQ(p.mag(), 3.0);
	vec3d o{};
	vec3d q{0.0, 4.0, 3.0};
	EXPECT_DOUBLE_EQ(o.dst(q), 5.0);
	EXPECT_DOUBLE_EQ(q.dot(p), 12.0);

	vec3d c{1.0, 0.0, 0.0};
	c.crs(vec3d{0.0, 1.0, 0.0});
	EXPECT_TRUE((c == vec3d{0.0, 0.0, 1.0}));

	vec3d n{2.0, 0.0, 0.0};
	n.nor();
	EXPECT_TRUE((n == vec3d{1.0, 0.0, 0.0}));
	EXPECT_DOUBLE_EQ(n.mag(), 1.0);

	vec3d z{};
	z.nor();
	EXPECT_TRUE((z == vec3d{0.0, 0.0, 0.0}));
	vec3d u{0.0, 1.0, 0.0};
	u.nor();
	EXPECT_TRUE((u == vec3d{0.0, 1.0, 0.0}));

	vec3d t{1e200, 1e200, 0.0};
	t.nor();
	EXPECT_DOUBLE_EQ(t.x, 0.0);
	EXPECT_DOUBLE_EQ(t.y, 0.0);
	EXPECT_DOUBLE_EQ(t.z, 0.0);
}

TEST(Vec3I, Layout) {
	using vec3i = bench_box::math::vec3i;
#if defined(BENCH_BOX_SIMD_REG4I32)
	static_assert(sizeof(vec3i) == 16);
	static_assert(alignof(vec3i) == 16);
#else
	static_assert(sizeof(vec3i) == 12);
	static_assert(alignof(vec3i) == alignof(std::int32_t));
#endif
}

TEST(Vec3I, Arithmetic) {
	using vec3i = bench_box::math::vec3i;

	static_assert(vec3i{1, 2, 3} + vec3i{4, 5, 6} == vec3i{5, 7, 9});
	static_assert(vec3i{4, 5, 6} - vec3i{1, 2, 3} == vec3i{3, 3, 3});
	static_assert(vec3i{1, 2, 3} * 2 == vec3i{2, 4, 6});
	static_assert(vec3i{1, 2, 3} / 2 == vec3i{0, 1, 1});
	static_assert(vec3i{1, 2, 3} + 1 == vec3i{2, 3, 4});
	static_assert(vec3i{1, 2, 3} - 1 == vec3i{0, 1, 2});
	static_assert(vec3i{2}.mag2() == 12);

	vec3i a{1, 2, 3};
	a += vec3i{1, 1, 1};
	EXPECT_TRUE((a == vec3i{2, 3, 4}));
	a -= vec3i{1, 1, 1};
	a *= 3;
	EXPECT_TRUE((a == vec3i{3, 6, 9}));
	a /= 3;
	EXPECT_TRUE((a == vec3i{1, 2, 3}));
	a += 1;
	EXPECT_TRUE((a == vec3i{2, 3, 4}));
	a -= 1;
	EXPECT_TRUE((a == vec3i{1, 2, 3}));
	EXPECT_TRUE((a != vec3i{1, 2, 4}));

	vec3i s{};
	s.set(7, -8, 9);
	EXPECT_TRUE((s == vec3i{7, -8, 9}));
	s.set(vec3i{1, 2, 3});
	EXPECT_TRUE((s == vec3i{1, 2, 3}));
}

TEST(Vec3I, Geometry) {
	using vec3i = bench_box::math::vec3i;

	static_assert(vec3i{1, 2, 3}.mag2() == 14);
	static_assert(vec3i{-7, -8, -9}.mag2() == 194);
	static_assert(vec3i{1, 2, 3}.dot(vec3i{4, 5, 6}) == 32);
	static_assert(vec3i{1, 2, 3}.dst2(vec3i{2, 2, 3}) == 1);

	vec3i p{0, 4, 3};
	EXPECT_EQ(p.mag2(), 25);
	vec3i o{};
	EXPECT_EQ(o.dst2(p), 25);

	vec3i c{1, 0, 0};
	c.crs(vec3i{0, 1, 0});
	EXPECT_TRUE((c == vec3i{0, 0, 1}));
}

TEST(Vec3LI, Layout) {
	using vec3li = bench_box::math::vec3li;
#if defined(BENCH_BOX_SIMD_REG4I64)
	static_assert(sizeof(vec3li) == 32);
	static_assert(alignof(vec3li) == 32);
#else
	static_assert(sizeof(vec3li) == 24);
	static_assert(alignof(vec3li) == alignof(std::int64_t));
#endif
}

TEST(Vec3LI, Arithmetic) {
	using vec3li = bench_box::math::vec3li;

	constexpr int64_t two = 2;
	constexpr int64_t three = 3;

	static_assert(vec3li{1, 2, 3} + vec3li{4, 5, 6} == vec3li{5, 7, 9});
	static_assert(vec3li{1, 2, 3} * two == vec3li{2, 4, 6});
	static_assert(vec3li{1, 2, 3} / two == vec3li{0, 1, 1});
	static_assert(vec3li{1, 2, 3} + three == vec3li{4, 5, 6});
	static_assert(vec3li{1, 2, 3} - three == vec3li{-2, -1, 0});
	static_assert(vec3li{2}.mag2() == 12);

	vec3li a{1, 2, 3};
	a += vec3li{1, 1, 1};
	EXPECT_TRUE((a == vec3li{2, 3, 4}));
	a -= vec3li{1, 1, 1};
	a *= 3;
	EXPECT_TRUE((a == vec3li{3, 6, 9}));
	a /= 3;
	EXPECT_TRUE((a == vec3li{1, 2, 3}));
	EXPECT_TRUE((a != vec3li{1, 2, 4}));

	vec3li s{};
	s.set(7, -8, 9);
	EXPECT_TRUE((s == vec3li{7, -8, 9}));
	s.set(vec3li{1, 2, 3});
	EXPECT_TRUE((s == vec3li{1, 2, 3}));
}

TEST(Vec3LI, Geometry) {
	using vec3li = bench_box::math::vec3li;

	static_assert(vec3li{100000, 200000, 300000}.mag2() == 140000000000);
	static_assert(vec3li{100000, 200000, 300000}.dot(vec3li{1, 2, 3}) == 1400000);
	static_assert(vec3li{0, 0, 0}.dst2(vec3li{100000, 200000, 300000}) == 140000000000);

	vec3li p{100000, 200000, 300000};
	EXPECT_EQ(p.mag2(), 140000000000);
	vec3li c{1, 0, 0};
	c.crs(vec3li{0, 100000, 0});
	EXPECT_TRUE((c == vec3li{0, 0, 100000}));
}

TEST(Vec3F, MinMaxClamp) {
	using bench_box::math::clamp;
	using bench_box::math::max;
	using bench_box::math::min;
	using vec3f = bench_box::math::vec3f;

	static_assert(min(vec3f{1.0f, 4.0f, 2.0f}, vec3f{3.0f, 2.0f, 5.0f}) == vec3f{1.0f, 2.0f, 2.0f});
	static_assert(max(vec3f{1.0f, 4.0f, 2.0f}, vec3f{3.0f, 2.0f, 5.0f}) == vec3f{3.0f, 4.0f, 5.0f});
	static_assert(clamp(vec3f{-1.0f, 5.0f, 2.0f}, vec3f{0.0f, 0.0f, 0.0f}, vec3f{3.0f, 3.0f, 3.0f}) == vec3f{0.0f, 3.0f, 2.0f});
	static_assert(clamp(vec3f{1.0f, 2.0f, 3.0f}, vec3f{0.0f, 0.0f, 0.0f}, vec3f{4.0f, 4.0f, 4.0f}) == vec3f{1.0f, 2.0f, 3.0f});
	static_assert(clamp(vec3f{-1.0f, 5.0f, 2.0f}, 0.0f, 3.0f) == vec3f{0.0f, 3.0f, 2.0f});
}

TEST(Vec3I, MinMaxClamp) {
	using bench_box::math::clamp;
	using bench_box::math::max;
	using bench_box::math::min;
	using vec3i = bench_box::math::vec3i;

	static_assert(min(vec3i{1, 4, 2}, vec3i{3, 2, 5}) == vec3i{1, 2, 2});
	static_assert(max(vec3i{1, 4, 2}, vec3i{3, 2, 5}) == vec3i{3, 4, 5});
	static_assert(clamp(vec3i{-1, 5, 2}, vec3i{0, 0, 0}, vec3i{3, 3, 3}) == vec3i{0, 3, 2});
	static_assert(clamp(vec3i{1, 2, 3}, vec3i{0, 0, 0}, vec3i{4, 4, 4}) == vec3i{1, 2, 3});
	static_assert(clamp(vec3i{-1, 5, 2}, 0, 3) == vec3i{0, 3, 2});
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
