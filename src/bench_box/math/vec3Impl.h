#pragma once
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::vec3(T scalar) noexcept : x(scalar), y(scalar), z(scalar) {
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::vec3(T nx, T ny, T nz) noexcept : x(nx), y(ny), z(nz) {
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec3& vector) noexcept {
			x = vector.x;
			y = vector.y;
			z = vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(T nx, T ny, T nz) noexcept {
			x = nx;
			y = ny;
			z = nz;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag2() const noexcept {
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if constexpr (std::is_same_v<T, double>) {
				if (!std::is_constant_evaluated()) {
					simd::reg2d rx = simd::load1d2d_lo(x);
					simd::reg2d ry = simd::load1d2d_lo(y);
					simd::reg2d rz = simd::load1d2d_lo(z);
					simd::reg2d i = simd::mul2d(ry, ry);
					i = simd::fmadd2d_lo(rx, rx, i);
					i = simd::fmadd2d_lo(rz, rz, i);
					double out = 0;
					simd::store2d_lo(out, i);
					return static_cast<T>(out);
				}
			}
		#endif
			return x * x + y * y + z * z;
		}

		template <typename T, std::size_t A>
		inline T vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag() const noexcept {
			return static_cast<T>(std::sqrt(mag2()));
		}

		template <typename T, std::size_t A>
		inline vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::nor() noexcept {
			const T length2 = mag2();
			if (length2 != static_cast<T>(0) && length2 != static_cast<T>(1)) {
				const T length = static_cast<T>(std::sqrt(length2));
				x /= length;
				y /= length;
				z /= length;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dst2(const vec3& vector) const noexcept {
			return (*this - vector).mag2();
		}

		template <typename T, std::size_t A>
		inline T vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dst(const vec3& vector) const noexcept {
			return static_cast<T>(std::sqrt(dst2(vector)));
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dot(const vec3& vector) const noexcept {
			return vector.x * x + vector.y * y + vector.z * z;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::crs(const vec3& vector) noexcept {
			T cx = y * vector.z - z * vector.y;
			T cy = z * vector.x - x * vector.z;
			T cz = x * vector.y - y * vector.x;
			x = cx;
			y = cy;
			z = cz;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr bool vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const vec3& vector) const noexcept {
			return (x == vector.x && y == vector.y && z == vector.z);
		}

		template <typename T, std::size_t A>
		constexpr bool vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const vec3& vector) const noexcept {
			return (x != vector.x || y != vector.y || z != vector.z);
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const vec3& vector) noexcept {
			x += vector.x;
			y += vector.y;
			z += vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const vec3& vector) noexcept {
			x -= vector.x;
			y -= vector.y;
			z -= vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(T value) noexcept {
			x += value;
			y += value;
			z += value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(T value) noexcept {
			x -= value;
			y -= value;
			z -= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(T value) noexcept {
			x *= value;
			y *= value;
			z *= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& vec3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator/=(T value) noexcept {
			x /= value;
			y /= value;
			z /= value;
			return *this;
		}

		template <>
		inline vec3d& vec3d::nor() noexcept {
			const double length2 = mag2();
			if (length2 != 0.0 && length2 != 1.0) {
			#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
				static_assert(alignof(vec3d) >= alignof(simd::reg4d), "vec3d must be aligned for SIMD loads");
				simd::store4d(x, simd::div4d(simd::load4d4d(x), simd::load1d4d(std::sqrt(length2))));
			#else
				const double length = std::sqrt(length2);
				x /= length;
				y /= length;
				z /= length;
			#endif
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::vec3(T scalar) noexcept : x(scalar), y(scalar), z(scalar) {
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::vec3(T nx, T ny, T nz) noexcept : x(nx), y(ny), z(nz) {
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::set(const vec3& vector) noexcept {
			x = vector.x;
			y = vector.y;
			z = vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::set(T nx, T ny, T nz) noexcept {
			x = nx;
			y = ny;
			z = nz;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::mag2() const noexcept {
			return x * x + y * y + z * z;
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::dst2(const vec3& vector) const noexcept {
			return (*this - vector).mag2();
		}

		template <typename T, std::size_t A>
		constexpr T vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::dot(const vec3& vector) const noexcept {
			return vector.x * x + vector.y * y + vector.z * z;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::crs(const vec3& vector) noexcept {
			T cx = y * vector.z - z * vector.y;
			T cy = z * vector.x - x * vector.z;
			T cz = x * vector.y - y * vector.x;
			x = cx;
			y = cy;
			z = cz;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr bool vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator==(const vec3& vector) const noexcept {
			return (x == vector.x && y == vector.y && z == vector.z);
		}

		template <typename T, std::size_t A>
		constexpr bool vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator!=(const vec3& vector) const noexcept {
			return (x != vector.x || y != vector.y || z != vector.z);
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator+=(const vec3& vector) noexcept {
			x += vector.x;
			y += vector.y;
			z += vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator-=(const vec3& vector) noexcept {
			x -= vector.x;
			y -= vector.y;
			z -= vector.z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator+=(T value) noexcept {
			x += value;
			y += value;
			z += value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator-=(T value) noexcept {
			x -= value;
			y -= value;
			z -= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator*=(T value) noexcept {
			x *= value;
			y *= value;
			z *= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>& vec3<T, A, std::enable_if_t<std::is_integral<T>::value>>::operator/=(T value) noexcept {
			x /= value;
			y /= value;
			z /= value;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator+(const vec3<T, A>& left, const vec3<T, A>& right) noexcept {
			return {left.x + right.x, left.y + right.y, left.z + right.z};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator-(const vec3<T, A>& left, const vec3<T, A>& right) noexcept {
			return {left.x - right.x, left.y - right.y, left.z - right.z};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator+(const vec3<T, A>& left, T value) noexcept {
			return {left.x + value, left.y + value, left.z + value};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator-(const vec3<T, A>& left, T value) noexcept {
			return {left.x - value, left.y - value, left.z - value};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator*(const vec3<T, A>& left, T value) noexcept {
			return {left.x * value, left.y * value, left.z * value};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> operator/(const vec3<T, A>& left, T value) noexcept {
			return {left.x / value, left.y / value, left.z / value};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> min(const vec3<T, A>& left, const vec3<T, A>& right) noexcept {
			return {
				left.x < right.x ? left.x : right.x,
				left.y < right.y ? left.y : right.y,
				left.z < right.z ? left.z : right.z
			};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> max(const vec3<T, A>& left, const vec3<T, A>& right) noexcept {
			return {
				left.x > right.x ? left.x : right.x,
				left.y > right.y ? left.y : right.y,
				left.z > right.z ? left.z : right.z
			};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> clamp(const vec3<T, A>& vector, const vec3<T, A>& lowest, const vec3<T, A>& highest) noexcept {
			return {
				vector.x < lowest.x ? lowest.x : (highest.x < vector.x ? highest.x : vector.x),
				vector.y < lowest.y ? lowest.y : (highest.y < vector.y ? highest.y : vector.y),
				vector.z < lowest.z ? lowest.z : (highest.z < vector.z ? highest.z : vector.z)
			};
		}

		template <typename T, std::size_t A>
		constexpr vec3<T, A> clamp(const vec3<T, A>& vector, T lowest, T highest) noexcept {
			return {
				vector.x < lowest ? lowest : (highest < vector.x ? highest : vector.x),
				vector.y < lowest ? lowest : (highest < vector.y ? highest : vector.y),
				vector.z < lowest ? lowest : (highest < vector.z ? highest : vector.z)
			};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
