#include <gtest/gtest.h>
#include <bench_box/math/mat4.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

using mat4f = bench_box::math::mat4f;
using mat4d = bench_box::math::mat4d;
using quatd = bench_box::math::quatd;
using quatf = bench_box::math::quatf;
using vec3d = bench_box::math::vec3d;
using vec3f = bench_box::math::vec3f;

constexpr double pi = 3.14159265358979323846;

TEST(Mat4F, Layout) {
	static_assert(sizeof(mat4f) == 64);
	static_assert(alignof(mat4f) == 64);
	static_assert(sizeof(mat4d) == 128);
	static_assert(alignof(mat4d) == 128);
	static_assert(std::is_nothrow_default_constructible_v<mat4d>);
	static_assert(std::is_nothrow_copy_constructible_v<mat4d>);
	static_assert(std::is_nothrow_move_constructible_v<mat4d>);
	static_assert(std::is_nothrow_copy_assignable_v<mat4d>);
	static_assert(std::is_nothrow_move_assignable_v<mat4d>);
	static_assert(std::is_nothrow_destructible_v<mat4d>);
}

TEST(Mat4F, DefaultIsIdentity) {
	static_assert(mat4f{} == mat4f{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
	static_assert(mat4d{} == mat4d{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});

	mat4f cleared{2.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
	EXPECT_TRUE((cleared.setIdentity() == mat4f{}));

	mat4d cleared_double{2.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0};
	EXPECT_TRUE((cleared_double.setIdentity() == mat4d{}));

	constexpr mat4f constexpr_identity_check{4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	static_assert(constexpr_identity_check.values[3] == 0.0f);
}

TEST(Mat4F, ArrayConstructor) {
	static constexpr float data[16] = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	static constexpr mat4f from_array{data};
	static_assert(from_array == mat4f{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	});

	mat4f assigned{};
	assigned.set(data);
	EXPECT_TRUE((assigned == from_array));
	EXPECT_EQ(assigned.values[3], 4.0f);
	EXPECT_EQ(assigned.values[15], 16.0f);
}

TEST(Mat4F, AddSubtract) {
	constexpr mat4f a{
		1.0f, 0.0f, 9.0f, 0.0f,
		2.0f, 0.0f, 0.0f, 6.0f,
		0.0f, 0.0f, 5.0f, 0.0f,
		4.0f, 0.0f, 0.0f, 1.0f
	};
	constexpr mat4f b{
		8.0f, 0.0f, 9.0f, 2.0f,
		2.0f, 3.0f, 0.0f, 4.0f,
		0.0f, 4.0f, 9.0f, 2.0f,
		4.0f, 6.0f, 0.0f, 1.0f
	};
	static_assert(a + b == mat4f{
		9.0f, 0.0f, 18.0f, 2.0f,
		4.0f, 3.0f, 0.0f, 10.0f,
		0.0f, 4.0f, 14.0f, 2.0f,
		8.0f, 6.0f, 0.0f, 2.0f
	});
	static_assert(b - a == mat4f{
		7.0f, 0.0f, 0.0f, 2.0f,
		0.0f, 3.0f, 0.0f, -2.0f,
		0.0f, 4.0f, 4.0f, 2.0f,
		0.0f, 6.0f, 0.0f, 0.0f
	});

	mat4f c = a;
	c += b;
	EXPECT_TRUE((c == a + b));
	c -= b;
	EXPECT_TRUE((c == a));
	c *= mat4f{};
	EXPECT_TRUE((c == a));
	EXPECT_TRUE((mat4d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0} != mat4d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0}));
}

TEST(Mat4F, Multiply) {
	constexpr mat4f a{
		1.0f, 0.0f, 9.0f, 0.0f,
		2.0f, 0.0f, 0.0f, 6.0f,
		0.0f, 0.0f, 5.0f, 0.0f,
		4.0f, 0.0f, 0.0f, 1.0f
	};
	constexpr mat4f b{
		8.0f, 0.0f, 9.0f, 2.0f,
		2.0f, 3.0f, 0.0f, 4.0f,
		0.0f, 4.0f, 9.0f, 2.0f,
		4.0f, 6.0f, 0.0f, 1.0f
	};
	static_assert(a * b == mat4f{
		8.0f, 36.0f, 90.0f, 20.0f,
		40.0f, 36.0f, 18.0f, 10.0f,
		0.0f, 20.0f, 45.0f, 10.0f,
		36.0f, 6.0f, 36.0f, 9.0f
	});

	constexpr mat4d identity_product = mat4d{
		9.0, 1.0, 3.0, 7.0,
		5.0, 9.0, 6.0, 1.0,
		3.0, 7.0, 3.0, 0.0,
		6.0, 2.0, 8.0, 3.0
	};
	static_assert(identity_product * mat4d{} == identity_product);
	static_assert(mat4d{} * identity_product == identity_product);

	mat4f compound{a};
	compound *= b;
	EXPECT_TRUE((compound == a * b));
}

TEST(Mat4F, Determinant) {
	static_assert(mat4f{}.determinant() == 1.0f);
	static_assert(mat4d{}.determinant() == 1.0);

	constexpr mat4f a{
		9.0f, 1.0f, 3.0f, 7.0f,
		5.0f, 9.0f, 6.0f, 1.0f,
		3.0f, 7.0f, 3.0f, 0.0f,
		6.0f, 2.0f, 8.0f, 3.0f
	};
	static_assert(a.determinant() == 152.0f);

	constexpr mat4d b{
		2.0, 0.0, 2.0, 0.0,
		2.0, 0.0, 0.0, 2.0,
		0.0, 0.0, 2.0, 0.0,
		2.0, 2.0, 0.0, 2.0
	};
	static_assert(b.determinant() == -16.0);

	mat4d transposed = b;
	transposed.transpose();
	EXPECT_DOUBLE_EQ(transposed.determinant(), b.determinant());

	const mat4d singular{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0
	};
	EXPECT_DOUBLE_EQ(singular.determinant(), 0.0);
}

TEST(Mat4F, Invert) {
	constexpr mat4f source{
		2.0f, 0.0f, 2.0f, 0.0f,
		2.0f, 0.0f, 0.0f, 2.0f,
		0.0f, 0.0f, 2.0f, 0.0f,
		2.0f, 2.0f, 0.0f, 2.0f
	};
	static_assert(source.determinant() == -16.0f);

	constexpr mat4f inverted = [&source]() noexcept {
		mat4f copy{source};
		copy.invert();
		return copy;
	}();
	static_assert(inverted == mat4f{
		0.5f, 0.0f, -0.5f, 0.0f,
		0.0f, -0.5f, 0.0f, 0.5f,
		0.0f, 0.0f, 0.5f, 0.0f,
		-0.5f, 0.5f, 0.5f, 0.0f
	});
	static_assert(source * inverted == mat4f{});

	mat4d unchanged{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0
	};
	EXPECT_DOUBLE_EQ(unchanged.determinant(), 0.0);
	unchanged.invert();
	const mat4d still_singular{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0
	};
	EXPECT_TRUE((unchanged == still_singular));

	mat4d wide{
		9.0, 1.0, 3.0, 7.0,
		5.0, 9.0, 6.0, 1.0,
		3.0, 7.0, 3.0, 0.0,
		6.0, 2.0, 8.0, 3.0
	};
	const mat4d wide_copy{wide};
	wide.invert();
	const mat4d product = wide_copy * wide;
	for (std::size_t i = 0; i < 16; i++) {
		const double expected = (i % 4 == i / 4) ? 1.0 : 0.0;
		EXPECT_NEAR(product.values[i], expected, 1e-12);
	}
}

TEST(Mat4F, Transpose) {
	mat4f a{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	EXPECT_TRUE((a.transpose() == mat4f{
		1.0f, 5.0f, 9.0f, 13.0f,
		2.0f, 6.0f, 10.0f, 14.0f,
		3.0f, 7.0f, 11.0f, 15.0f,
		4.0f, 8.0f, 12.0f, 16.0f
	}));
	a.transpose();
	EXPECT_TRUE((a == mat4f{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	}));

	constexpr mat4d doubled = []() noexcept {
		mat4d copy{
			1.0, 5.0, 9.0, 13.0,
			2.0, 6.0, 10.0, 14.0,
			3.0, 7.0, 11.0, 15.0,
			4.0, 8.0, 12.0, 16.0
		};
		copy.transpose();
		return copy;
	}();
	static_assert(doubled == mat4d{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0
	});
	EXPECT_TRUE((mat4f{}.transpose() == mat4f{}));
}

TEST(Mat4D, Translation) {
	static_assert(mat4d{vec3d{1.0, 2.0, 3.0}, quatd{}, vec3d{1.0, 1.0, 1.0}} == mat4d{
		1.0, 0.0, 0.0, 1.0,
		0.0, 1.0, 0.0, 2.0,
		0.0, 0.0, 1.0, 3.0,
		0.0, 0.0, 0.0, 1.0
	});

	const mat4d scaled{vec3d{0.0, 0.0, 0.0}, quatd{}, vec3d{2.0, 3.0, 4.0}};
	EXPECT_DOUBLE_EQ(scaled.determinant(), 24.0);
	EXPECT_TRUE(((scaled * vec3d{1.0, 1.0, 1.0}) == vec3d{2.0, 3.0, 4.0}));
	EXPECT_TRUE(((vec3d{1.0, 1.0, 1.0} * scaled) == vec3d{2.0, 3.0, 4.0}));

	mat4d built{vec3d{1.0, 2.0, 3.0}, quatd{}, vec3d{2.0, 3.0, 4.0}};
	mat4d set{};
	set.set(vec3d{1.0, 2.0, 3.0}, quatd{}, vec3d{2.0, 3.0, 4.0});
	EXPECT_TRUE((built == set));

	const mat4d translation{vec3d{5.0, -6.0, 7.0}, quatd{}, vec3d{1.0, 1.0, 1.0}};
	EXPECT_TRUE(((translation * vec3d{1.0, 1.0, 1.0}) == vec3d{6.0, -5.0, 8.0}));
}

TEST(Mat4D, Rotation) {
	constexpr mat4d half_turn{
		-2.0, 0.0, 0.0, 1.0,
		0.0, -3.0, 0.0, 2.0,
		0.0, 0.0, 4.0, 3.0,
		0.0, 0.0, 0.0, 1.0
	};
	static_assert(mat4d{vec3d{1.0, 2.0, 3.0}, quatd{0.0, 0.0, 1.0, 0.0}, vec3d{2.0, 3.0, 4.0}} == half_turn);
	static_assert(half_turn.determinant() == 24.0);

	const quatd half_turn_quat{vec3d{0.0, 0.0, 1.0}, pi};
	const mat4d half_turn_by_set = [&half_turn_quat]() noexcept {
		mat4d built{};
		built.set(vec3d{1.0, 2.0, 3.0}, half_turn_quat, vec3d{2.0, 3.0, 4.0});
		return built;
	}();
	const vec3d half_turn_point = half_turn_by_set * vec3d{1.0, 1.0, 1.0};
	EXPECT_NEAR(half_turn_point.x, -1.0, 1e-12);
	EXPECT_NEAR(half_turn_point.y, -1.0, 1e-12);
	EXPECT_NEAR(half_turn_point.z, 7.0, 1e-12);

	const quatd around_z{vec3d{0.0, 0.0, 1.0}, pi / 2.0};
	const mat4d rotated{vec3d{0.0, 0.0, 0.0}, around_z, vec3d{1.0, 1.0, 1.0}};
	const vec3d rotated_vector = rotated * vec3d{1.0, 0.0, 0.0};
	EXPECT_NEAR(rotated_vector.x, 0.0, 1e-12);
	EXPECT_NEAR(rotated_vector.y, 1.0, 1e-12);
	EXPECT_NEAR(rotated_vector.z, 0.0, 1e-12);
	EXPECT_NEAR(rotated.determinant(), 1.0, 1e-12);

	const vec3d by_quat = around_z * vec3d{1.0, 0.0, 0.0};
	EXPECT_NEAR(rotated_vector.x, by_quat.x, 1e-15);
	EXPECT_NEAR(rotated_vector.y, by_quat.y, 1e-15);
	EXPECT_NEAR(rotated_vector.z, by_quat.z, 1e-15);

	const mat4d scaled_rotation{vec3d{1.0, 2.0, 3.0}, around_z, vec3d{2.0, 3.0, 4.0}};
	const vec3d point = scaled_rotation * vec3d{1.0, 1.0, 1.0};
	const vec3d scaled_parts = around_z * vec3d{2.0, 3.0, 4.0} + vec3d{1.0, 2.0, 3.0};
	EXPECT_NEAR(point.x, scaled_parts.x, 1e-12);
	EXPECT_NEAR(point.y, scaled_parts.y, 1e-12);
	EXPECT_NEAR(point.z, scaled_parts.z, 1e-12);
	EXPECT_NEAR(point.x, -2.0, 1e-12);
	EXPECT_NEAR(point.y, 4.0, 1e-12);
	EXPECT_NEAR(point.z, 7.0, 1e-12);
	EXPECT_NEAR(scaled_rotation.determinant(), 24.0, 1e-12);

	mat4d inverse{scaled_rotation};
	inverse.invert();
	const vec3d back = inverse * (scaled_rotation * vec3d{0.5, -0.25, 2.0});
	EXPECT_NEAR(back.x, 0.5, 1e-12);
	EXPECT_NEAR(back.y, -0.25, 1e-12);
	EXPECT_NEAR(back.z, 2.0, 1e-12);

	const mat4d identity_product = scaled_rotation * inverse;
	for (std::size_t i = 0; i < 16; i++) {
		const double expected = (i % 4 == i / 4) ? 1.0 : 0.0;
		EXPECT_NEAR(identity_product.values[i], expected, 1e-12);
	}

	const quatd around_y{vec3d{0.0, 1.0, 0.0}, pi / 2.0};
	const mat4d spinning{vec3d{0.0, 0.0, 0.0}, around_y, vec3d{1.0, 1.0, 1.0}};
	const vec3d spun = spinning * vec3d{0.0, 0.0, 1.0};
	const vec3d by_spinning_quat = around_y * vec3d{0.0, 0.0, 1.0};
	EXPECT_NEAR(spun.x, by_spinning_quat.x, 1e-15);
	EXPECT_NEAR(spun.y, by_spinning_quat.y, 1e-15);
	EXPECT_NEAR(spun.z, by_spinning_quat.z, 1e-15);
	EXPECT_NEAR(spun.x, 1.0, 1e-12);
	EXPECT_NEAR(spun.y, 0.0, 1e-12);
	EXPECT_NEAR(spun.z, 0.0, 1e-12);

	const mat4d composed = scaled_rotation * mat4d{};
	EXPECT_TRUE((composed == scaled_rotation));
}

TEST(Mat4D, View) {
	const mat4d view{vec3d{0.0, -1.0, 0.0}, vec3d{0.0, 0.0, 1.0}};
	const mat4d expected_view{
		-1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	};
	EXPECT_TRUE((view == expected_view));

	const vec3d forward = view * vec3d{0.0, -1.0, 0.0};
	EXPECT_NEAR(forward.x, 0.0, 1e-12);
	EXPECT_NEAR(forward.y, 0.0, 1e-12);
	EXPECT_NEAR(forward.z, -1.0, 1e-12);

	const vec3d up = view * vec3d{0.0, 0.0, 1.0};
	EXPECT_NEAR(up.x, 0.0, 1e-12);
	EXPECT_NEAR(up.y, 1.0, 1e-12);
	EXPECT_NEAR(up.z, 0.0, 1e-12);

	const vec3d origin = view * vec3d{1.0, 2.0, 3.0};
	const vec3d swapped = view * vec3d{2.0, 3.0, 1.0};
	EXPECT_TRUE((origin == vec3d{-1.0, 3.0, 2.0}));
	EXPECT_TRUE((swapped == vec3d{-2.0, 1.0, 3.0}));

	const mat4d side{vec3d{-1.0, 0.0, 0.0}, vec3d{0.0, 1.0, 0.0}};
	const vec3d side_forward = side * vec3d{-1.0, 0.0, 0.0};
	const vec3d side_up = side * vec3d{0.0, 1.0, 0.0};
	const vec3d side_right = side * vec3d{0.0, 0.0, 1.0};
	EXPECT_NEAR(side_forward.x, 0.0, 1e-12);
	EXPECT_NEAR(side_forward.y, 0.0, 1e-12);
	EXPECT_NEAR(side_forward.z, -1.0, 1e-12);
	EXPECT_NEAR(side_up.x, 0.0, 1e-12);
	EXPECT_NEAR(side_up.y, 1.0, 1e-12);
	EXPECT_NEAR(side_up.z, 0.0, 1e-12);
	EXPECT_NEAR(side_right.x, -1.0, 1e-12);
	EXPECT_NEAR(side_right.y, 0.0, 1e-12);
	EXPECT_NEAR(side_right.z, 0.0, 1e-12);

	mat4d along_negative_z{vec3d{0.0, 0.0, -1.0}, vec3d{0.0, 1.0, 0.0}};
	const mat4d along_view{vec3d{0.0, 0.0, 0.0}, quatd{}, vec3d{1.0, 1.0, 1.0}};
	const vec3d along_forward = along_negative_z * vec3d{0.0, 0.0, -1.0};
	const vec3d along_up = along_negative_z * vec3d{0.0, 1.0, 0.0};
	EXPECT_NEAR(along_forward.x, 0.0, 1e-12);
	EXPECT_NEAR(along_forward.y, 0.0, 1e-12);
	EXPECT_NEAR(along_forward.z, -1.0, 1e-12);
	EXPECT_NEAR(along_up.x, 0.0, 1e-12);
	EXPECT_NEAR(along_up.y, 1.0, 1e-12);
	EXPECT_NEAR(along_up.z, 0.0, 1e-12);
	EXPECT_TRUE((along_negative_z == along_view));

	mat4d set_view{};
	set_view.set(vec3d{0.0, -1.0, 0.0}, vec3d{0.0, 0.0, 1.0});
	EXPECT_TRUE((set_view == view));

	mat4d transposed_view{view};
	transposed_view.transpose();
	const mat4d orthogonal = view * transposed_view;
	for (std::size_t i = 0; i < 16; i++) {
		const double expected = (i % 4 == i / 4) ? 1.0 : 0.0;
		EXPECT_NEAR(orthogonal.values[i], expected, 1e-12);
	}
	EXPECT_NEAR(view.determinant(), 1.0, 1e-12);
}

TEST(Mat4D, Transform) {
	constexpr mat4d a{
		1.0, 0.0, 9.0, 2.0,
		2.0, 0.0, 0.0, 6.0,
		0.0, 0.0, 5.0, 0.0,
		4.0, 0.0, 0.0, 1.0
	};
	static_assert((a * vec3d{1.0, 1.0, 1.0}) == vec3d{12.0, 8.0, 5.0});
	static_assert((vec3d{1.0, 1.0, 1.0} * a) == vec3d{12.0, 8.0, 5.0});

	constexpr mat4d b{
		2.0, 0.0, 0.0, 1.0,
		0.0, 3.0, 0.0, 2.0,
		0.0, 0.0, 4.0, 3.0,
		0.0, 0.0, 0.0, 1.0
	};
	static_assert((b * vec3d{1.0, 1.0, 1.0}) == vec3d{3.0, 5.0, 7.0});

	constexpr mat4d composed = a * b;
	static_assert(composed == mat4d{
		2.0, 0.0, 36.0, 30.0,
		4.0, 0.0, 0.0, 8.0,
		0.0, 0.0, 20.0, 15.0,
		8.0, 0.0, 0.0, 5.0
	});
	static_assert((composed * vec3d{1.0, 1.0, 1.0}) == vec3d{68.0, 12.0, 35.0});

	constexpr vec3d vector{0.5, -0.25, 2.0};
	const vec3d sequential = composed * vector;
	const vec3d by_parts = a * (b * vector);
	static_assert((composed * vector) == vec3d{103.0, 10.0, 55.0});
	EXPECT_DOUBLE_EQ(sequential.x, by_parts.x);
	EXPECT_DOUBLE_EQ(sequential.y, by_parts.y);
	EXPECT_DOUBLE_EQ(sequential.z, by_parts.z);

	const mat4d identity_transform{};
	EXPECT_TRUE(((identity_transform * vec3d{4.0, 5.0, 6.0}) == vec3d{4.0, 5.0, 6.0}));
	EXPECT_TRUE(((vec3d{4.0, 5.0, 6.0} * identity_transform) == vec3d{4.0, 5.0, 6.0}));

	constexpr auto transposed = [](const mat4d source) noexcept {
		mat4d copy{source};
		copy.transpose();
		return copy;
	};
	static_assert(transposed(mat4d{vec3d{1.0, 2.0, 3.0}, quatd{}, vec3d{1.0, 1.0, 1.0}}) == mat4d{
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		1.0, 2.0, 3.0, 1.0
	});

	constexpr mat4d scaling{vec3d{0.0, 0.0, 0.0}, quatd{}, vec3d{2.0, 3.0, 4.0}};
	static_assert((scaling * vec3d{1.0, 2.0, 3.0}) == vec3d{2.0, 6.0, 12.0});
	static_assert((vec3d{1.0, 2.0, 3.0} * scaling) == vec3d{2.0, 6.0, 12.0});
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
