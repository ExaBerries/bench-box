#pragma once
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
		template <typename T, std::size_t A = vec3_alignment<T>, typename = void>
		struct alignas(A) mat3;

		template <typename T, std::size_t A>
		struct alignas(A) mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>> {
			static_assert(A >= alignof(T), "mat3 alignment must be at least the element alignment");

			using element_t = T;

			// rows are padded for simd
			T values[12] = {
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0
			};

			[[nodiscard]] constexpr T* operator[](std::size_t row) noexcept {
				return values + row * 4;
			}

			[[nodiscard]] constexpr const T* operator[](std::size_t row) const noexcept {
				return values + row * 4;
			}

			[[nodiscard]] constexpr T& at(std::size_t row, std::size_t column) noexcept {
				return values[row * 4 + column];
			}

			[[nodiscard]] constexpr const T& at(std::size_t row, std::size_t column) const noexcept {
				return values[row * 4 + column];
			}

			mat3() = default;
			explicit constexpr mat3(const T (&values)[9]) noexcept;
			constexpr mat3(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8) noexcept;
			mat3(const mat3&) = default;
			mat3(mat3&&) noexcept = default;
			~mat3() = default;

			constexpr mat3& setIdentity() noexcept;
			constexpr mat3& set(const T (&values)[9]) noexcept;
			template <std::size_t VA, std::size_t VB>
			constexpr mat3& setOuter(const vec3<T, VA>& left, const vec3<T, VB>& right) noexcept;

			[[nodiscard]] constexpr T determinant() const noexcept;
			constexpr mat3& invert() noexcept;
			constexpr mat3& transpose() noexcept;

			[[nodiscard]] constexpr bool operator==(const mat3& matrix) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const mat3& matrix) const noexcept;

			constexpr mat3& operator+=(const mat3& matrix) noexcept;
			constexpr mat3& operator-=(const mat3& matrix) noexcept;
			constexpr mat3& operator*=(const mat3& matrix) noexcept;

			mat3& operator=(const mat3&) = default;
			mat3& operator=(mat3&&) noexcept = default;
		};

		using mat3f = mat3<float>;
		using mat3d = mat3<double>;

		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator+(const mat3<T, A>& left, const mat3<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator-(const mat3<T, A>& left, const mat3<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator*(const mat3<T, A>& left, const mat3<T, A>& right) noexcept;

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const mat3<T, A>& matrix, const vec3<T, VA>& vector) noexcept;
		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const vec3<T, VA>& vector, const mat3<T, A>& matrix) noexcept;

		// Dyadic product: (left * right^T)[i][j] == left[i] * right[j].
		template <typename T, std::size_t VA, std::size_t VB, std::size_t A = vec3_alignment<T>>
		[[nodiscard]] constexpr mat3<T, A> outer(const vec3<T, VA>& left, const vec3<T, VB>& right) noexcept;
	} // namespace math
} // namespace bench_box

#include "mat3Impl.h"
