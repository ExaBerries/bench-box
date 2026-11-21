#pragma once
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::quat(T nx, T ny, T nz, T nw) noexcept : x(nx), y(ny), z(nz), w(nw) {
		}

		template <typename T, std::size_t A>
		template <std::size_t VA>
		inline quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::quat(const vec3<T, VA>& axis, T angle) noexcept {
			const T half = angle / static_cast<T>(2);
			const T sine = static_cast<T>(std::sin(half));
			x = axis.x * sine;
			y = axis.y * sine;
			z = axis.z * sine;
			w = static_cast<T>(std::cos(half));
			nor();
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const quat& quaternion) noexcept {
			x = quaternion.x;
			y = quaternion.y;
			z = quaternion.z;
			w = quaternion.w;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(T nx, T ny, T nz, T nw) noexcept {
			x = nx;
			y = ny;
			z = nz;
			w = nw;
			return *this;
		}

		template <typename T, std::size_t A>
		template <std::size_t VA>
		inline quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec3<T, VA>& axis, T angle) noexcept {
			const T half = angle / static_cast<T>(2);
			const T sine = static_cast<T>(std::sin(half));
			x = axis.x * sine;
			y = axis.y * sine;
			z = axis.z * sine;
			w = static_cast<T>(std::cos(half));
			return nor();
		}

		template <typename T, std::size_t A>
		constexpr T quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag2() const noexcept {
			return x * x + y * y + z * z + w * w;
		}

		template <typename T, std::size_t A>
		inline T quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mag() const noexcept {
			return static_cast<T>(std::sqrt(mag2()));
		}

		template <typename T, std::size_t A>
		inline quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::nor() noexcept {
			const T length2 = mag2();
			if (length2 != static_cast<T>(0) && length2 != static_cast<T>(1)) {
				const T length = static_cast<T>(std::sqrt(length2));
				x /= length;
				y /= length;
				z /= length;
				w /= length;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::conjugate() noexcept {
			x = -x;
			y = -y;
			z = -z;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::dot(const quat& quaternion) const noexcept {
			return x * quaternion.x + y * quaternion.y + z * quaternion.z + w * quaternion.w;
		}

		template <typename T, std::size_t A>
		constexpr bool quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const quat& quaternion) const noexcept {
			return (x == quaternion.x && y == quaternion.y && z == quaternion.z && w == quaternion.w);
		}

		template <typename T, std::size_t A>
		constexpr bool quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const quat& quaternion) const noexcept {
			return (x != quaternion.x || y != quaternion.y || z != quaternion.z || w != quaternion.w);
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const quat& quaternion) noexcept {
			x += quaternion.x;
			y += quaternion.y;
			z += quaternion.z;
			w += quaternion.w;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const quat& quaternion) noexcept {
			x -= quaternion.x;
			y -= quaternion.y;
			z -= quaternion.z;
			w -= quaternion.w;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& quat<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(const quat& quaternion) noexcept {
			const T tx = w * quaternion.x + x * quaternion.w + y * quaternion.z - z * quaternion.y;
			const T ty = w * quaternion.y + y * quaternion.w + z * quaternion.x - x * quaternion.z;
			const T tz = w * quaternion.z + z * quaternion.w + x * quaternion.y - y * quaternion.x;
			const T tw = w * quaternion.w - x * quaternion.x - y * quaternion.y - z * quaternion.z;
			x = tx;
			y = ty;
			z = tz;
			w = tw;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A> operator+(const quat<T, A>& left, const quat<T, A>& right) noexcept {
			return {left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w};
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A> operator-(const quat<T, A>& left, const quat<T, A>& right) noexcept {
			return {left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w};
		}

		template <typename T, std::size_t A>
		constexpr quat<T, A> operator*(const quat<T, A>& left, const quat<T, A>& right) noexcept {
			return {
				left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
				left.w * right.y + left.y * right.w + left.z * right.x - left.x * right.z,
				left.w * right.z + left.z * right.w + left.x * right.y - left.y * right.x,
				left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
			};
		}

		template <typename T, std::size_t A, std::size_t VA>
		inline vec3<T, VA> operator*(const quat<T, A>& quaternion, const vec3<T, VA>& vector) noexcept {
			const quat<T, A> vector_as_quat(vector.x, vector.y, vector.z, static_cast<T>(0));
			const quat<T, A> rotated = quaternion * vector_as_quat * quat<T, A>(quaternion).conjugate();
			return {rotated.x, rotated.y, rotated.z};
		}

		template <typename T, std::size_t A, std::size_t VA>
		inline vec3<T, VA> operator*(const vec3<T, VA>& vector, const quat<T, A>& quaternion) noexcept {
			const quat<T, A> vector_as_quat(vector.x, vector.y, vector.z, static_cast<T>(0));
			const quat<T, A> rotated = quaternion * vector_as_quat * quat<T, A>(quaternion).conjugate();
			return {rotated.x, rotated.y, rotated.z};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
