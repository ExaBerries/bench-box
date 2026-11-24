#pragma once
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::vec2(T scalar) noexcept : x(scalar), y(scalar) {
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::vec2(T nx, T ny) noexcept : x(nx), y(ny) {
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec2& vector) noexcept {
			x = vector.x;
			y = vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(T nx, T ny) noexcept {
			x = nx;
			y = ny;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag2() const noexcept {
			return x * x + y * y;
		}

		template <typename T, std::size_t A>
		inline T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag() const noexcept {
			return static_cast<T>(std::sqrt(mag2()));
		}

		template <typename T, std::size_t A>
		inline vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::nor() noexcept {
			const T length2 = mag2();
			if (length2 != static_cast<T>(0) && length2 != static_cast<T>(1)) {
				const T length = static_cast<T>(std::sqrt(length2));
				x /= length;
				y /= length;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dst2(const vec2& vector) const noexcept {
			return (*this - vector).mag2();
		}

		template <typename T, std::size_t A>
		inline T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dst(const vec2& vector) const noexcept {
			return static_cast<T>(std::sqrt(dst2(vector)));
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dot(const vec2& vector) const noexcept {
			return vector.x * x + vector.y * y;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::crs(const vec2& vector) const noexcept {
			return x * vector.y - y * vector.x;
		}

		template <typename T, std::size_t A>
		constexpr bool vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const vec2& vector) const noexcept {
			return (x == vector.x && y == vector.y);
		}

		template <typename T, std::size_t A>
		constexpr bool vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const vec2& vector) const noexcept {
			return (x != vector.x || y != vector.y);
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const vec2& vector) noexcept {
			x += vector.x;
			y += vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const vec2& vector) noexcept {
			x -= vector.x;
			y -= vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(T value) noexcept {
			x += value;
			y += value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(T value) noexcept {
			x -= value;
			y -= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(T value) noexcept {
			x *= value;
			y *= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec2<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator/=(T value) noexcept {
			x /= value;
			y /= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::vec2(T scalar) noexcept : x(scalar), y(scalar) {
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::vec2(T nx, T ny) noexcept : x(nx), y(ny) {
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::set(const vec2& vector) noexcept {
			x = vector.x;
			y = vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::set(T nx, T ny) noexcept {
			x = nx;
			y = ny;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::mag2() const noexcept {
			return x * x + y * y;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::dst2(const vec2& vector) const noexcept {
			return (*this - vector).mag2();
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::dot(const vec2& vector) const noexcept {
			return vector.x * x + vector.y * y;
		}

		template <typename T, std::size_t A>
		constexpr T vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::crs(const vec2& vector) const noexcept {
			return x * vector.y - y * vector.x;
		}

		template <typename T, std::size_t A>
		constexpr bool vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator==(const vec2& vector) const noexcept {
			return (x == vector.x && y == vector.y);
		}

		template <typename T, std::size_t A>
		constexpr bool vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator!=(const vec2& vector) const noexcept {
			return (x != vector.x || y != vector.y);
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator+=(const vec2& vector) noexcept {
			x += vector.x;
			y += vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator-=(const vec2& vector) noexcept {
			x -= vector.x;
			y -= vector.y;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator+=(T value) noexcept {
			x += value;
			y += value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator-=(T value) noexcept {
			x -= value;
			y -= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator*=(T value) noexcept {
			x *= value;
			y *= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec2<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator/=(T value) noexcept {
			x /= value;
			y /= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator+(const vec2<T, A>& left, const vec2<T, A>& right) noexcept {
			return {left.x + right.x, left.y + right.y};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator-(const vec2<T, A>& left, const vec2<T, A>& right) noexcept {
			return {left.x - right.x, left.y - right.y};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator+(const vec2<T, A>& left, T value) noexcept {
			return {left.x + value, left.y + value};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator-(const vec2<T, A>& left, T value) noexcept {
			return {left.x - value, left.y - value};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator*(const vec2<T, A>& left, T value) noexcept {
			return {left.x * value, left.y * value};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> operator/(const vec2<T, A>& left, T value) noexcept {
			return {left.x / value, left.y / value};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> min(const vec2<T, A>& left, const vec2<T, A>& right) noexcept {
			return {
				left.x < right.x ? left.x : right.x,
				left.y < right.y ? left.y : right.y
			};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> max(const vec2<T, A>& left, const vec2<T, A>& right) noexcept {
			return {
				left.x > right.x ? left.x : right.x,
				left.y > right.y ? left.y : right.y
			};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> clamp(const vec2<T, A>& vector, const vec2<T, A>& lowest, const vec2<T, A>& highest) noexcept {
			return {
				vector.x < lowest.x ? lowest.x : (highest.x < vector.x ? highest.x : vector.x),
				vector.y < lowest.y ? lowest.y : (highest.y < vector.y ? highest.y : vector.y)
			};
		}

		template <typename T, std::size_t A>
		constexpr vec2<T, A> clamp(const vec2<T, A>& vector, T lowest, T highest) noexcept {
			return {
				vector.x < lowest ? lowest : (highest < vector.x ? highest : vector.x),
				vector.y < lowest ? lowest : (highest < vector.y ? highest : vector.y)
			};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
