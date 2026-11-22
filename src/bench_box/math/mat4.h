#pragma once
#include <bench_box/math/quat.h>
#include <bench_box/math/vec3.h>
#include <cmath>
#include <cstddef>
#include <type_traits>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A = 16 * sizeof(T), typename = void>
		struct alignas(A) mat4;

		template <typename T, std::size_t A>
		struct alignas(A) mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>> {
			static_assert(A >= 64, "mat4 needs an alignment of at least 64 bytes for the simd loads");

			using element_t = T;

			T values[16] = {
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			};

			mat4() = default;
			constexpr explicit mat4(const T (&values)[16]) noexcept;
			template <std::size_t VA, std::size_t QA>
			constexpr mat4(const vec3<T, VA>& translation, const quat<T, QA>& rotation, const vec3<T, VA>& scale) noexcept;
			template <std::size_t VA>
			inline mat4(const vec3<T, VA>& direction, const vec3<T, VA>& up) noexcept;
			constexpr mat4(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept;
			mat4(const mat4&) = default;
			mat4(mat4&&) noexcept = default;
			~mat4() = default;

			constexpr mat4& setIdentity() noexcept;
			constexpr mat4& set(const T (&values)[16]) noexcept;
			template <std::size_t VA, std::size_t QA>
			constexpr mat4& set(const vec3<T, VA>& translation, const quat<T, QA>& rotation, const vec3<T, VA>& scale) noexcept;
			template <std::size_t VA>
			inline mat4& set(const vec3<T, VA>& direction, const vec3<T, VA>& up) noexcept;

			[[nodiscard]] constexpr T determinant() const noexcept;
			constexpr mat4& invert() noexcept;
			constexpr mat4& transpose() noexcept;

			[[nodiscard]] constexpr bool operator==(const mat4& matrix) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const mat4& matrix) const noexcept;

			constexpr mat4& operator+=(const mat4& matrix) noexcept;
			constexpr mat4& operator-=(const mat4& matrix) noexcept;
			constexpr mat4& operator*=(const mat4& matrix) noexcept;

			mat4& operator=(const mat4&) = default;
			mat4& operator=(mat4&&) noexcept = default;
		};

		using mat4f = mat4<float>;
		using mat4d = mat4<double>;

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator+(const mat4<T, A>& left, const mat4<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator-(const mat4<T, A>& left, const mat4<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator*(const mat4<T, A>& left, const mat4<T, A>& right) noexcept;

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const mat4<T, A>& matrix, const vec3<T, VA>& vector) noexcept;
		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const vec3<T, VA>& vector, const mat4<T, A>& matrix) noexcept;
	} // namespace math
} // namespace bench_box

#include "mat4Impl.h"
