#include <gtest/gtest.h>
#include <bench_box/math/mat3.h>
#include <cstddef>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

using mat3d = bench_box::math::mat3d;
using mat3f = bench_box::math::mat3f;
using vec3d = bench_box::math::vec3d;
using vec3f = bench_box::math::vec3f;

TEST(Mat3F, Layout) {
	static_assert(sizeof(mat3f) == 12 * sizeof(float));
	static_assert(sizeof(mat3d) == 12 * sizeof(double));
	static_assert(alignof(mat3f) == alignof(vec3f));
	static_assert(alignof(mat3d) == alignof(vec3d));
	static_assert(std::is_nothrow_default_constructible_v<mat3d>);
	static_assert(std::is_nothrow_copy_constructible_v<mat3d>);
	static_assert(std::is_nothrow_move_constructible_v<mat3d>);
	static_assert(std::is_nothrow_copy_assignable_v<mat3d>);
	static_assert(std::is_nothrow_move_assignable_v<mat3d>);
	static_assert(std::is_nothrow_destructible_v<mat3d>);
}

TEST(Mat3F, DefaultIsIdentity) {
	static_assert(mat3f{} == mat3f{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});
	static_assert(mat3d{} == mat3d{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

	mat3f cleared{2.0f, 7.0f, 6.0f, 4.0f, 3.0f, 2.0f, 9.0f, 8.0f, 7.0f};
	EXPECT_TRUE((cleared.setIdentity() == mat3f{}));
	EXPECT_EQ(cleared.values[3], 0.0f);
	EXPECT_EQ(cleared.values[7], 0.0f);
	EXPECT_EQ(cleared.values[11], 0.0f);

	constexpr mat3f constexpr_identity_check{4.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	static_assert(constexpr_identity_check.values[3] == 0.0f);
}

TEST(Mat3F, ArrayConstructor) {
	static constexpr float data[9] = {
		1.0f, 2.0f, 3.0f,
		5.0f, 6.0f, 7.0f,
		9.0f, 10.0f, 11.0f
	};
	static constexpr mat3f from_array{data};
	static_assert(from_array == mat3f{
		1.0f, 2.0f, 3.0f,
		5.0f, 6.0f, 7.0f,
		9.0f, 10.0f, 11.0f
	});

	mat3f assigned{};
	assigned.set(data);
	EXPECT_TRUE((assigned == from_array));
	EXPECT_EQ(assigned.values[0], 1.0f);
	EXPECT_EQ(assigned.values[10], 11.0f);
	EXPECT_EQ(assigned.values[3], 0.0f);
	EXPECT_EQ(assigned.values[7], 0.0f);
	EXPECT_EQ(assigned.values[11], 0.0f);
}

TEST(Mat3F, RowColumnAccess) {
	static constexpr mat3f matrix{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	static_assert(matrix[0][0] == 1.0f);
	static_assert(matrix[1][2] == 6.0f);
	static_assert(matrix[2][2] == 9.0f);
	static_assert(matrix.at(1, 2) == 6.0f);
	static_assert(matrix.at(2, 0) == 7.0f);

	mat3f writable = matrix;
	writable[1][2] = 42.0f;
	EXPECT_EQ(writable.values[6], 42.0f);
	writable.at(2, 0) = 24.0f;
	EXPECT_EQ(writable.values[8], 24.0f);

	const mat3f& view = writable;
	EXPECT_EQ(view[1][2], 42.0f);
	EXPECT_EQ(view.at(2, 0), 24.0f);
}

TEST(Mat3F, OuterProduct) {
	constexpr vec3f a{1.0f, 2.0f, 3.0f};
	constexpr vec3f b{4.0f, 5.0f, 6.0f};
	constexpr mat3f ab{
		4.0f, 5.0f, 6.0f,
		8.0f, 10.0f, 12.0f,
		12.0f, 15.0f, 18.0f
	};
	constexpr mat3f ba{
		4.0f, 8.0f, 12.0f,
		5.0f, 10.0f, 15.0f,
		6.0f, 12.0f, 18.0f
	};

	static_assert(outer(a, b) == ab);
	static_assert(outer(b, a) == ba);
	static_assert(outer(a, b) != outer(b, a));

	mat3f built;
	EXPECT_TRUE((built.setOuter(a, b) == ab));
	EXPECT_EQ(built.values[3], 0.0f);
	EXPECT_EQ(built.values[7], 0.0f);
	EXPECT_EQ(built.values[11], 0.0f);

	constexpr vec3d ad{1.0, 2.0, 3.0};
	constexpr vec3d bd{4.0, 5.0, 6.0};
	static_assert(outer(ad, bd) == mat3d{
		4.0, 5.0, 6.0,
		8.0, 10.0, 12.0,
		12.0, 15.0, 18.0
	});

	// (a outer b) * v == a * (b . v)
	constexpr vec3f v{1.0f, 1.0f, 1.0f};
	const vec3f left_side = outer(a, b) * v;
	const vec3f right_side = a * b.dot(v);
	EXPECT_NEAR(left_side.x, right_side.x, 1e-6f);
	EXPECT_NEAR(left_side.y, right_side.y, 1e-6f);
	EXPECT_NEAR(left_side.z, right_side.z, 1e-6f);
}

TEST(Mat3F, AddSubtract) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	constexpr mat3f b{
		1.0f, 0.0f, 2.0f,
		0.0f, 1.0f, 0.0f,
		3.0f, 0.0f, 1.0f
	};
	static_assert(a + b == mat3f{
		2.0f, 2.0f, 5.0f,
		0.0f, 2.0f, 4.0f,
		8.0f, 6.0f, 1.0f
	});
	static_assert(b - a == mat3f{
		0.0f, -2.0f, -1.0f,
		0.0f, 0.0f, -4.0f,
		-2.0f, -6.0f, 1.0f
	});

	mat3f c = a;
	c += b;
	EXPECT_TRUE((c == a + b));
	c -= b;
	EXPECT_TRUE((c == a));
	c *= mat3f{};
	EXPECT_TRUE((c == a));
	EXPECT_TRUE((mat3d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0} != mat3d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0}));
}

TEST(Mat3F, Multiply) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	constexpr mat3f b{
		1.0f, 0.0f, 2.0f,
		0.0f, 1.0f, 0.0f,
		3.0f, 0.0f, 1.0f
	};
	static_assert(a * b == mat3f{
		10.0f, 2.0f, 5.0f,
		12.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 10.0f
	});

	constexpr mat3d identity_product = mat3d{
		9.0, 1.0, 3.0,
		5.0, 9.0, 6.0,
		3.0, 7.0, 3.0
	};
	static_assert(identity_product * mat3d{} == identity_product);
	static_assert(mat3d{} * identity_product == identity_product);

	mat3f compound{a};
	compound *= b;
	EXPECT_TRUE((compound == a * b));

	// padding lanes must stay zero after the SIMD and scalar paths
	const mat3f product = b * a;
	EXPECT_EQ(product.values[3], 0.0f);
	EXPECT_EQ(product.values[7], 0.0f);
	EXPECT_EQ(product.values[11], 0.0f);
}

TEST(Mat3F, VectorMultiply) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	constexpr vec3f v{1.0f, 2.0f, 3.0f};

	static_assert(a * v == vec3f{14.0f, 14.0f, 17.0f});
	static_assert(v * a == vec3f{16.0f, 22.0f, 11.0f});

	const vec3f column = a * v;
	EXPECT_NEAR(column.x, 14.0f, 1e-6);
	EXPECT_NEAR(column.y, 14.0f, 1e-6);
	EXPECT_NEAR(column.z, 17.0f, 1e-6);

	const vec3f row = v * a;
	EXPECT_NEAR(row.x, 16.0f, 1e-6);
	EXPECT_NEAR(row.y, 22.0f, 1e-6);
	EXPECT_NEAR(row.z, 11.0f, 1e-6);

	constexpr mat3d ad{
		1.0, 2.0, 3.0,
		0.0, 1.0, 4.0,
		5.0, 6.0, 0.0
	};
	constexpr vec3d vd{1.0, 2.0, 3.0};
	static_assert(ad * vd == vec3d{14.0, 14.0, 17.0});
	static_assert(vd * ad == vec3d{16.0, 22.0, 11.0});
}

TEST(Mat3F, Transpose) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	constexpr mat3f transposed{
		1.0f, 0.0f, 5.0f,
		2.0f, 1.0f, 6.0f,
		3.0f, 4.0f, 0.0f
	};

	mat3f runtime{a};
	runtime.transpose();
	EXPECT_TRUE((runtime == transposed));
	EXPECT_EQ(runtime.values[3], 0.0f);
	EXPECT_EQ(runtime.values[7], 0.0f);
	EXPECT_EQ(runtime.values[11], 0.0f);

	runtime.transpose();
	EXPECT_TRUE((runtime == a));
}

TEST(Mat3F, Determinant) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	static_assert(a.determinant() == 1.0f);
	static_assert(mat3f{}.determinant() == 1.0f);
	static_assert(mat3f{2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f}.determinant() == 24.0f);

	constexpr mat3d b{
		4.0, 3.0, 2.0,
		1.0, 5.0, 7.0,
		8.0, 6.0, 9.0
	};
	static_assert(b.determinant() == 85.0);
}

TEST(Mat3F, Invert) {
	constexpr mat3f a{
		1.0f, 2.0f, 3.0f,
		0.0f, 1.0f, 4.0f,
		5.0f, 6.0f, 0.0f
	};
	constexpr mat3f expected{
		-24.0f, 18.0f, 5.0f,
		20.0f, -15.0f, -4.0f,
		-5.0f, 4.0f, 1.0f
	};

	const mat3f inverse = mat3f{a}.invert();
	for (std::size_t row = 0; row < 3; row++) {
		for (std::size_t column = 0; column < 3; column++) {
			EXPECT_NEAR(inverse.at(row, column), expected.at(row, column), 1e-5f);
		}
	}
	EXPECT_EQ(inverse.values[3], 0.0f);
	EXPECT_EQ(inverse.values[7], 0.0f);
	EXPECT_EQ(inverse.values[11], 0.0f);

	const mat3f product = a * inverse;
	for (std::size_t row = 0; row < 3; row++) {
		for (std::size_t column = 0; column < 3; column++) {
			const float want = row == column ? 1.0f : 0.0f;
			EXPECT_NEAR(product.at(row, column), want, 1e-5f);
		}
	}

	// singular matrices are left untouched, matching mat4
	const mat3f singular{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	};
	EXPECT_TRUE((mat3f{singular}.invert() == singular));
}
