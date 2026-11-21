#include <gtest/gtest.h>
#include <bench_box/math/quat.h>
#include <cmath>
#include <type_traits>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

using quatd = bench_box::math::quatd;
using quatf = bench_box::math::quatf;
using vec3d = bench_box::math::vec3d;
using vec3f = bench_box::math::vec3f;

constexpr double pi = 3.14159265358979323846;

template <typename Q>
concept floating_point_quat = requires(Q q) {
	{ q.mag() };
	{ q.nor() };
};

static_assert(floating_point_quat<quatf>);
static_assert(floating_point_quat<quatd>);
static_assert(std::is_same<quatd::element_t, double>::value);
static_assert(std::is_same<quatf::element_t, float>::value);

TEST(QuatF, Layout) {
	static_assert(sizeof(quatf) == 16);
	static_assert(alignof(quatf) == 16);
	static_assert(sizeof(quatd) == 32);
	static_assert(alignof(quatd) == 32);
	static_assert(std::is_nothrow_default_constructible_v<quatd>);
	static_assert(std::is_nothrow_copy_constructible_v<quatd>);
	static_assert(std::is_nothrow_move_constructible_v<quatd>);
	static_assert(std::is_nothrow_copy_assignable_v<quatd>);
	static_assert(std::is_nothrow_move_assignable_v<quatd>);
}

TEST(QuatF, DefaultIsIdentity) {
	static_assert(quatf{} == quatf{0.0f, 0.0f, 0.0f, 1.0f});
	static_assert(quatd{} == quatd{0.0, 0.0, 0.0, 1.0});
	static_assert(quatd{}.mag2() == 1.0);
	static_assert(quatf{}.mag2() == 1.0f);
}

TEST(QuatF, Arithmetic) {
	constexpr quatd a{1.0, 2.0, 3.0, 4.0};
	constexpr quatd b{5.0, 6.0, 7.0, 8.0};
	static_assert(a + b == quatd{6.0, 8.0, 10.0, 12.0});
	static_assert(b - a == quatd{4.0, 4.0, 4.0, 4.0});
	static_assert(a.mag2() == 30.0);
	static_assert(a.dot(b) == 70.0);
	static_assert(a != b);

	quatd conjugated{1.0, 2.0, 3.0, 4.0};
	EXPECT_TRUE((conjugated.conjugate() == quatd{-1.0, -2.0, -3.0, 4.0}));
	conjugated.conjugate();
	EXPECT_TRUE((conjugated == a));

	quatd c{1.0, 1.0, 1.0, 1.0};
	c += a;
	EXPECT_TRUE((c == quatd{2.0, 3.0, 4.0, 5.0}));
	c -= a;
	EXPECT_TRUE((c == quatd{1.0, 1.0, 1.0, 1.0}));

	quatd s{};
	s.set(1.0, 2.0, 3.0, 4.0);
	EXPECT_TRUE((s == a));
	s.set(b);
	EXPECT_TRUE((s == b));

	quatd doubled{a};
	doubled += a;
	EXPECT_TRUE((doubled == quatd{2.0, 4.0, 6.0, 8.0}));
}

TEST(QuatF, Normalize) {
	quatd identity{};
	EXPECT_TRUE((identity.nor() == quatd{}));

	quatd zero{};
	zero.set(0.0, 0.0, 0.0, 0.0);
	zero.nor();
	EXPECT_TRUE((zero == quatd{0.0, 0.0, 0.0, 0.0}));

	quatd a{1.0, 2.0, 3.0, 4.0};
	a.nor();
	EXPECT_DOUBLE_EQ(a.mag(), 1.0);
	EXPECT_DOUBLE_EQ(a.x, 1.0 / std::sqrt(30.0));
	EXPECT_DOUBLE_EQ(a.y, 2.0 / std::sqrt(30.0));
	EXPECT_DOUBLE_EQ(a.z, 3.0 / std::sqrt(30.0));
	EXPECT_DOUBLE_EQ(a.w, 4.0 / std::sqrt(30.0));

	quatd b{2.0, 0.0, 0.0, 0.0};
	b.nor();
	EXPECT_TRUE((b == quatd{1.0, 0.0, 0.0, 0.0}));

	quatd unit{0.5, 0.5, 0.5, 0.5};
	unit.nor();
	EXPECT_TRUE((unit == quatd{0.5, 0.5, 0.5, 0.5}));
	static_assert(quatd{0.5, 0.5, 0.5, 0.5}.mag2() == 1.0);
}

TEST(QuatF, Multiply) {
	constexpr quatd x{1.0, 0.0, 0.0, 0.0};
	constexpr quatd y{0.0, 1.0, 0.0, 0.0};
	constexpr quatd z{0.0, 0.0, 1.0, 0.0};
	static_assert(x * y == z);
	static_assert(y * x == quatd{0.0, 0.0, -1.0, 0.0});
	static_assert(x * quatd{} == x);
	static_assert(quatd{} * y == y);
	static_assert(z * y == quatd{-1.0, 0.0, 0.0, 0.0});
	static_assert(x * x == quatd{0.0, 0.0, 0.0, -1.0});
	static_assert(y * y == quatd{0.0, 0.0, 0.0, -1.0});
	static_assert(z * z == quatd{0.0, 0.0, 0.0, -1.0});
	static_assert(x * y * z == quatd{0.0, 0.0, 0.0, -1.0});
	static_assert(z * x == quatd{0.0, 1.0, 0.0, 0.0});

	quatd a{0.0, 0.0, 1.0, 0.0};
	a *= y;
	EXPECT_TRUE((a == z * y));

	constexpr quatd a2{1.0, 2.0, 3.0, 4.0};
	constexpr quatd b2{5.0, 6.0, 7.0, 8.0};
	static_assert(a2 * b2 == quatd{24.0, 48.0, 48.0, -6.0});
}

TEST(QuatD, AxisAngle) {
	const quatd no_rotation{vec3d{0.0, 0.0, 1.0}, 0.0};
	EXPECT_TRUE((no_rotation == quatd{}));

	const quatd around_z{vec3d{0.0, 0.0, 1.0}, pi / 2.0};
	EXPECT_NEAR(around_z.mag(), 1.0, 1e-15);
	EXPECT_NEAR(around_z.x, 0.0, 1e-15);
	EXPECT_NEAR(around_z.y, 0.0, 1e-15);
	EXPECT_NEAR(around_z.z, std::sqrt(0.5), 1e-15);
	EXPECT_NEAR(around_z.w, std::sqrt(0.5), 1e-15);

	const quatd unnormalized_axis{vec3d{0.0, 2.0, 0.0}, pi / 2.0};
	EXPECT_NEAR(unnormalized_axis.mag(), 1.0, 1e-15);
	EXPECT_NEAR(unnormalized_axis.x, 0.0, 1e-15);
	EXPECT_NEAR(unnormalized_axis.y, std::sqrt(0.8), 1e-15);
	EXPECT_NEAR(unnormalized_axis.z, 0.0, 1e-15);
	EXPECT_NEAR(unnormalized_axis.w, std::sqrt(0.2), 1e-15);

	quatd set_back{};
	set_back.set(vec3d{0.0, 0.0, 1.0}, pi / 2.0);
	EXPECT_TRUE((set_back == around_z));

	const quatd half_turn{vec3d{1.0, 0.0, 0.0}, pi};
	EXPECT_NEAR(half_turn.x, 1.0, 1e-15);
	EXPECT_NEAR(half_turn.y, 0.0, 1e-15);
	EXPECT_NEAR(half_turn.z, 0.0, 1e-15);
	EXPECT_NEAR(half_turn.w, 0.0, 1e-15);
}

TEST(QuatD, RotateVector) {
	const quatd around_z{vec3d{0.0, 0.0, 1.0}, pi / 2.0};
	const quatd around_x{vec3d{1.0, 0.0, 0.0}, pi / 2.0};
	const quatd around_y{vec3d{0.0, 1.0, 0.0}, pi / 2.0};
	const quatd half_turn{vec3d{0.0, 0.0, 1.0}, pi};

	const vec3d x_to_y = around_z * vec3d{1.0, 0.0, 0.0};
	EXPECT_NEAR(x_to_y.x, 0.0, 1e-15);
	EXPECT_NEAR(x_to_y.y, 1.0, 1e-15);
	EXPECT_NEAR(x_to_y.z, 0.0, 1e-15);

	const vec3d y_to_x = vec3d{1.0, 0.0, 0.0} * around_z;
	EXPECT_NEAR(y_to_x.x, 0.0, 1e-15);
	EXPECT_NEAR(y_to_x.y, 1.0, 1e-15);
	EXPECT_NEAR(y_to_x.z, 0.0, 1e-15);

	const vec3d y_to_z = around_x * vec3d{0.0, 1.0, 0.0};
	EXPECT_NEAR(y_to_z.x, 0.0, 1e-15);
	EXPECT_NEAR(y_to_z.y, 0.0, 1e-15);
	EXPECT_NEAR(y_to_z.z, 1.0, 1e-15);

	const vec3d z_to_x = around_y * vec3d{0.0, 0.0, 1.0};
	EXPECT_NEAR(z_to_x.x, 1.0, 1e-15);
	EXPECT_NEAR(z_to_x.y, 0.0, 1e-15);
	EXPECT_NEAR(z_to_x.z, 0.0, 1e-15);

	const vec3d along_axis = around_y * vec3d{0.0, 3.0, 0.0};
	EXPECT_NEAR(along_axis.x, 0.0, 1e-15);
	EXPECT_NEAR(along_axis.y, 3.0, 1e-15);
	EXPECT_NEAR(along_axis.z, 0.0, 1e-15);

	const vec3d flipped = half_turn * vec3d{1.0, 2.0, 0.0};
	EXPECT_NEAR(flipped.x, -1.0, 1e-15);
	EXPECT_NEAR(flipped.y, -2.0, 1e-15);
	EXPECT_NEAR(flipped.z, 0.0, 1e-15);

	const quatd spin{vec3d{1.0, 1.0, 1.0}, pi / 3.0};
	const vec3d point{0.3, -0.7, 0.5};
	const vec3d spun = spin * point;
	EXPECT_NEAR(spun.x, 0.7261536563608846, 1e-15);
	EXPECT_NEAR(spun.y, -0.08213672050459198, 1e-15);
	EXPECT_NEAR(spun.z, -0.5440169358562925, 1e-15);
	EXPECT_NEAR(spin.mag(), 1.0, 1e-15);

	quatd spin_inverse{spin};
	const vec3d back = spin_inverse.conjugate() * spun;
	EXPECT_NEAR(back.x, point.x, 1e-15);
	EXPECT_NEAR(back.y, point.y, 1e-15);
	EXPECT_NEAR(back.z, point.z, 1e-15);

	const vec3d twice = spin * (point * 2.0);
	EXPECT_NEAR(twice.x, spun.x * 2.0, 1e-15);
	EXPECT_NEAR(twice.y, spun.y * 2.0, 1e-15);
	EXPECT_NEAR(twice.z, spun.z * 2.0, 1e-15);

	const vec3d untouched = quatd{} * point;
	EXPECT_TRUE((untouched == point));
	EXPECT_TRUE((point * quatd{} == point));
}

TEST(QuatD, Compose) {
	const quatd a{vec3d{0.0, 0.0, 1.0}, pi / 2.0};
	const quatd b{vec3d{0.0, 1.0, 0.0}, pi / 2.0};
	const quatd composed = a * b;
	const vec3d vector{1.0, 0.0, 0.0};

	const vec3d by_composed = composed * vector;
	const vec3d by_parts = a * (b * vector);
	EXPECT_NEAR(by_composed.x, by_parts.x, 1e-15);
	EXPECT_NEAR(by_composed.y, by_parts.y, 1e-15);
	EXPECT_NEAR(by_composed.z, by_parts.z, 1e-15);
	EXPECT_NEAR(by_composed.x, 0.0, 1e-15);
	EXPECT_NEAR(by_composed.y, 0.0, 1e-15);
	EXPECT_NEAR(by_composed.z, -1.0, 1e-15);

	EXPECT_NEAR(composed.x, -0.5, 1e-15);
	EXPECT_NEAR(composed.y, 0.5, 1e-15);
	EXPECT_NEAR(composed.z, 0.5, 1e-15);
	EXPECT_NEAR(composed.w, 0.5, 1e-15);
	EXPECT_NEAR(composed.mag(), 1.0, 1e-15);
	EXPECT_NEAR(composed.dot(composed), 1.0, 1e-15);

	const quatd doubled = composed + composed;
	EXPECT_NEAR(doubled.mag(), 2.0, 1e-15);
	const quatd zero = composed - composed;
	quatd rounded_zero{};
	rounded_zero.set(0.0, 0.0, 0.0, 0.0);
	EXPECT_TRUE((zero == rounded_zero));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
