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
		template <typename T, std::size_t A = 4 * sizeof(T), typename = void>
		struct alignas(A) quat;

		template <typename T, std::size_t A>
		struct alignas(A) quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>> {
			using element_t = T;

			T x = 0;
			T y = 0;
			T z = 0;
			T w = 1;

			quat() = default;
			constexpr quat(T x, T y, T z, T w) noexcept;
			template <std::size_t VA>
			inline quat(const vec3<T, VA>& axis, T angle) noexcept;
			quat(const quat&) = default;
			quat(quat&&) noexcept = default;
			~quat() = default;

			constexpr quat& set(const quat& quaternion) noexcept;
			constexpr quat& set(T x, T y, T z, T w) noexcept;
			template <std::size_t VA>
			inline quat& set(const vec3<T, VA>& axis, T angle) noexcept;

			[[nodiscard]] constexpr T mag2() const noexcept;
			[[nodiscard]] inline T mag() const noexcept;
			inline quat& nor() noexcept;
			constexpr quat& conjugate() noexcept;
			[[nodiscard]] constexpr T dot(const quat& quaternion) const noexcept;

			[[nodiscard]] constexpr bool operator==(const quat& quaternion) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const quat& quaternion) const noexcept;

			constexpr quat& operator+=(const quat& quaternion) noexcept;
			constexpr quat& operator-=(const quat& quaternion) noexcept;
			constexpr quat& operator*=(const quat& quaternion) noexcept;

			quat& operator=(const quat&) = default;
			quat& operator=(quat&&) noexcept = default;
		};

		using quatf = quat<float>;
		using quatd = quat<double>;

		template <typename T, std::size_t A>
		constexpr quat<T, A> operator+(const quat<T, A>& left, const quat<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr quat<T, A> operator-(const quat<T, A>& left, const quat<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr quat<T, A> operator*(const quat<T, A>& left, const quat<T, A>& right) noexcept;

		template <typename T, std::size_t A, std::size_t VA>
		inline vec3<T, VA> operator*(const quat<T, A>& quaternion, const vec3<T, VA>& vector) noexcept;
		template <typename T, std::size_t A, std::size_t VA>
		inline vec3<T, VA> operator*(const vec3<T, VA>& vector, const quat<T, A>& quaternion) noexcept;
	} // namespace math
} // namespace bench_box

#include "quatImpl.h"
