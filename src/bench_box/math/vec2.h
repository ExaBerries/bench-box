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
		inline constexpr std::size_t vec2_alignment = alignof(T);

#if defined(BENCH_BOX_SIMD_REG2D)
		template <>
		inline constexpr std::size_t vec2_alignment<double> = alignof(simd::reg2d);
#endif

#if defined(BENCH_BOX_SIMD_REG4F)
		template <>
		inline constexpr std::size_t vec2_alignment<float> = alignof(simd::reg4f);
#endif

#if defined(BENCH_BOX_SIMD_REG4I32)
		template <>
		inline constexpr std::size_t vec2_alignment<std::int32_t> = alignof(simd::reg4i32);
#endif

#if defined(BENCH_BOX_SIMD_REG2I64)
		template <>
		inline constexpr std::size_t vec2_alignment<std::int64_t> = alignof(simd::reg2i64);
#endif

		template <typename T, std::size_t A = vec2_alignment<T>, typename = void>
		struct alignas(A) vec2;

		template <typename T, std::size_t A>
		struct alignas(A) vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>> {
			using element_t = T;

			T x = 0;
			T y = 0;

			vec2() = default;
			constexpr vec2(T scalar) noexcept;
			constexpr vec2(T x, T y) noexcept;
			vec2(const vec2&) = default;
			vec2(vec2&&) noexcept = default;
			~vec2() = default;

			constexpr vec2& set(const vec2& vector) noexcept;
			constexpr vec2& set(T x, T y) noexcept;

			[[nodiscard]] constexpr T mag2() const noexcept;
			[[nodiscard]] inline T mag() const noexcept;
			inline vec2& nor() noexcept;
			[[nodiscard]] constexpr T dst2(const vec2& vector) const noexcept;
			[[nodiscard]] inline T dst(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr T dot(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr T crs(const vec2& vector) const noexcept;

			[[nodiscard]] constexpr bool operator==(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const vec2& vector) const noexcept;

			constexpr vec2& operator+=(const vec2& vector) noexcept;
			constexpr vec2& operator-=(const vec2& vector) noexcept;

			constexpr vec2& operator+=(T value) noexcept;
			constexpr vec2& operator-=(T value) noexcept;
			constexpr vec2& operator*=(T value) noexcept;
			constexpr vec2& operator/=(T value) noexcept;

			vec2& operator=(const vec2&) = default;
			vec2& operator=(vec2&&) noexcept = default;
		};

		template <typename T, std::size_t A>
		struct alignas(A) vec2<T, A, std::enable_if_t<std::is_integral<T>::value>> {
			using element_t = T;

			T x = 0;
			T y = 0;

			vec2() = default;
			constexpr vec2(T scalar) noexcept;
			constexpr vec2(T x, T y) noexcept;
			vec2(const vec2&) = default;
			vec2(vec2&&) noexcept = default;
			~vec2() = default;

			constexpr vec2& set(const vec2& vector) noexcept;
			constexpr vec2& set(T x, T y) noexcept;

			[[nodiscard]] constexpr T mag2() const noexcept;
			[[nodiscard]] constexpr T dst2(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr T dot(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr T crs(const vec2& vector) const noexcept;

			[[nodiscard]] constexpr bool operator==(const vec2& vector) const noexcept;
			[[nodiscard]] constexpr bool operator!=(const vec2& vector) const noexcept;

			constexpr vec2& operator+=(const vec2& vector) noexcept;
			constexpr vec2& operator-=(const vec2& vector) noexcept;

			constexpr vec2& operator+=(T value) noexcept;
			constexpr vec2& operator-=(T value) noexcept;
			constexpr vec2& operator*=(T value) noexcept;
			constexpr vec2& operator/=(T value) noexcept;

			vec2& operator=(const vec2&) = default;
			vec2& operator=(vec2&&) noexcept = default;
		};

		using vec2i = vec2<int32_t>;
		using vec2li = vec2<int64_t>;
		using vec2f = vec2<float>;
		using vec2d = vec2<double>;

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator+(const vec2<T, A>& left, const vec2<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator-(const vec2<T, A>& left, const vec2<T, A>& right) noexcept;

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator+(const vec2<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator-(const vec2<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator*(const vec2<T, A>& left, T value) noexcept;
		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator/(const vec2<T, A>& left, T value) noexcept;

		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec2<T, A> min(const vec2<T, A>& left, const vec2<T, A>& right) noexcept;
		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec2<T, A> max(const vec2<T, A>& left, const vec2<T, A>& right) noexcept;

		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec2<T, A> clamp(const vec2<T, A>& vector, const vec2<T, A>& lowest, const vec2<T, A>& highest) noexcept;
		template <typename T, std::size_t A>
		[[nodiscard]] constexpr vec2<T, A> clamp(const vec2<T, A>& vector, T lowest, T highest) noexcept;
	} // namespace math
} // namespace bench_box

#include "vec2Impl.h"
