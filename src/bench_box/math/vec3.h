#pragma once
#include <bench_box/isa/intrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
		template <typename T>
		inline constexpr std::size_t vec3_alignment = alignof(T);

#if defined(BENCH_BOX_SIMD_REG4D)
		template <>
		inline constexpr std::size_t vec3_alignment<double> = alignof(simd::reg4d);
#endif

#if defined(BENCH_BOX_SIMD_REG4F)
		template <>
		inline constexpr std::size_t vec3_alignment<float> = alignof(simd::reg4f);
#endif

#if defined(BENCH_BOX_SIMD_REG4I32)
		template <>
		inline constexpr std::size_t vec3_alignment<std::int32_t> = alignof(simd::reg4i32);
#endif

#if defined(BENCH_BOX_SIMD_REG4I64)
		template <>
		inline constexpr std::size_t vec3_alignment<std::int64_t> = alignof(simd::reg4i64);
#endif

		template <typename T, std::size_t A = vec3_alignment<T>, typename = void>
		struct alignas(A) vec3;

		template <typename T, std::size_t A>
		struct alignas(A) vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>> {
			using element_t = T;

			T x = 0;
			T y = 0;
			T z = 0;

			vec3() = default;
			constexpr vec3(T scalar) noexcept;
			constexpr vec3(T x, T y, T z) noexcept;
			vec3(const vec3&) = default;
			vec3(vec3&&) noexcept = default;
			~vec3() = default;

			constexpr vec3& set(const vec3& vector) noexcept;
			constexpr vec3& set(T x, T y, T z) noexcept;

			[[nodiscard]] constexpr T mag2() const noexcept;
			[[nodiscard]] inline T mag() const noexcept;
			inline vec3& nor() noexcept;
			[[nodiscard]] constexpr T dst2(const vec3& vector) const noexcept;
			[[nodiscard]] inline T dst(const vec3& vector) const noexcept;
			[[nodiscard]] constexpr T dot(const vec3& vector) const noexcept;
			constexpr vec3& crs(const vec3& vector) noexcept;

			[[nodiscard]] constexpr bool operator==(const vec3& vector) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const vec3& vector) const noexcept;

			constexpr vec3& operator+=(const vec3& vector) noexcept;
			constexpr vec3& operator-=(const vec3& vector) noexcept;

			constexpr vec3& operator+=(T value) noexcept;
			constexpr vec3& operator-=(T value) noexcept;
			constexpr vec3& operator*=(T value) noexcept;
			constexpr vec3& operator/=(T value) noexcept;

			vec3& operator=(const vec3&) = default;
			vec3& operator=(vec3&&) noexcept = default;
		};

		template <typename T, std::size_t A>
		struct alignas(A) vec3<T, A, std::enable_if_t<std::is_integral<T>::value>> {
			using element_t = T;

			T x = 0;
			T y = 0;
			T z = 0;

			vec3() = default;
			constexpr vec3(T scalar) noexcept;
			constexpr vec3(T x, T y, T z) noexcept;
			vec3(const vec3&) = default;
			vec3(vec3&&) noexcept = default;
			~vec3() = default;

			constexpr vec3& set(const vec3& vector) noexcept;
			constexpr vec3& set(T x, T y, T z) noexcept;

			[[nodiscard]] constexpr T mag2() const noexcept;
			[[nodiscard]] constexpr T dst2(const vec3& vector) const noexcept;
			[[nodiscard]] constexpr T dot(const vec3& vector) const noexcept;
			constexpr vec3& crs(const vec3& vector) noexcept;

			[[nodiscard]] constexpr bool operator==(const vec3& vector) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const vec3& vector) const noexcept;

			constexpr vec3& operator+=(const vec3& vector) noexcept;
			constexpr vec3& operator-=(const vec3& vector) noexcept;

			constexpr vec3& operator+=(T value) noexcept;
			constexpr vec3& operator-=(T value) noexcept;
			constexpr vec3& operator*=(T value) noexcept;
			constexpr vec3& operator/=(T value) noexcept;

			vec3& operator=(const vec3&) = default;
			vec3& operator=(vec3&&) noexcept = default;
		};

		using vec3i = vec3<int32_t>;
		using vec3li = vec3<int64_t>;
		using vec3f = vec3<float>;
		using vec3d = vec3<double>;

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator+(const vec3<T, A>& left, const vec3<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator-(const vec3<T, A>& left, const vec3<T, A>& right) noexcept;

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator+(const vec3<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator-(const vec3<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator*(const vec3<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator/(const vec3<T, A>& left, T value) noexcept;

		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec3<T, A> min(const vec3<T, A>& left, const vec3<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec3<T, A> max(const vec3<T, A>& left, const vec3<T, A>& right) noexcept;

		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec3<T, A> clamp(const vec3<T, A>& vector, const vec3<T, A>& lowest, const vec3<T, A>& highest) noexcept;
		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec3<T, A> clamp(const vec3<T, A>& vector, T lowest, T highest) noexcept;
	} // namespace math
} // namespace bench_box

#include "vec3Impl.h"
