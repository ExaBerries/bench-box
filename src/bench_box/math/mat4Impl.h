#pragma once
#include <cstddef>
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat4(const T (&nvalues)[16]) noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				values[i] = nvalues[i];
			}
		}

		template <typename T, std::size_t A>
		template <std::size_t VA, std::size_t QA>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat4(const vec3<T, VA>& translation, const quat<T, QA>& rotation, const vec3<T, VA>& scale) noexcept {
			set(translation, rotation, scale);
		}

		template <typename T, std::size_t A>
		template <std::size_t VA>
		inline mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat4(const vec3<T, VA>& direction, const vec3<T, VA>& up) noexcept {
			set(direction, up);
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat4(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept {
			values[0] = v0;
			values[1] = v1;
			values[2] = v2;
			values[3] = v3;
			values[4] = v4;
			values[5] = v5;
			values[6] = v6;
			values[7] = v7;
			values[8] = v8;
			values[9] = v9;
			values[10] = v10;
			values[11] = v11;
			values[12] = v12;
			values[13] = v13;
			values[14] = v14;
			values[15] = v15;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::setIdentity() noexcept {
			values[0] = 1;
			values[1] = 0;
			values[2] = 0;
			values[3] = 0;
			values[4] = 0;
			values[5] = 1;
			values[6] = 0;
			values[7] = 0;
			values[8] = 0;
			values[9] = 0;
			values[10] = 1;
			values[11] = 0;
			values[12] = 0;
			values[13] = 0;
			values[14] = 0;
			values[15] = 1;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const T (&nvalues)[16]) noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				values[i] = nvalues[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		template <std::size_t VA, std::size_t QA>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec3<T, VA>& translation, const quat<T, QA>& rotation, const vec3<T, VA>& scale) noexcept {
			const T xx = rotation.x * rotation.x;
			const T yy = rotation.y * rotation.y;
			const T zz = rotation.z * rotation.z;
			const T wx = rotation.w * rotation.x;
			const T wy = rotation.w * rotation.y;
			const T wz = rotation.w * rotation.z;
			values[0] = scale.x * (static_cast<T>(1) - (yy + zz) * 2);
			values[1] = scale.y * (rotation.x * rotation.y * 2 - wz * 2);
			values[2] = scale.z * (rotation.x * rotation.z * 2 + wy * 2);
			values[3] = translation.x;
			values[4] = scale.x * (rotation.x * rotation.y * 2 + wz * 2);
			values[5] = scale.y * (static_cast<T>(1) - (xx + zz) * 2);
			values[6] = scale.z * (rotation.y * rotation.z * 2 - wx * 2);
			values[7] = translation.y;
			values[8] = scale.x * (rotation.x * rotation.z * 2 - wy * 2);
			values[9] = scale.y * (rotation.y * rotation.z * 2 + wx * 2);
			values[10] = scale.z * (static_cast<T>(1) - (xx + yy) * 2);
			values[11] = translation.z;
			values[12] = 0;
			values[13] = 0;
			values[14] = 0;
			values[15] = 1;
			return *this;
		}

		template <typename T, std::size_t A>
		template <std::size_t VA>
		inline mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec3<T, VA>& direction, const vec3<T, VA>& up) noexcept {
			const vec3<T, VA> forward = vec3<T, VA>(direction).nor();
			const vec3<T, VA> right = vec3<T, VA>(forward).crs(up).nor();
			const vec3<T, VA> upward = vec3<T, VA>(right).crs(forward).nor();
			values[0] = right.x;
			values[1] = right.y;
			values[2] = right.z;
			values[3] = 0;
			values[4] = upward.x;
			values[5] = upward.y;
			values[6] = upward.z;
			values[7] = 0;
			values[8] = -forward.x;
			values[9] = -forward.y;
			values[10] = -forward.z;
			values[11] = 0;
			values[12] = 0;
			values[13] = 0;
			values[14] = 0;
			values[15] = 1;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::determinant() const noexcept {
			const T s0 = values[10] * values[15] - values[11] * values[14];
			const T s1 = values[9] * values[15] - values[11] * values[13];
			const T s2 = values[9] * values[14] - values[10] * values[13];
			const T s3 = values[8] * values[15] - values[11] * values[12];
			const T s4 = values[8] * values[14] - values[10] * values[12];
			const T s5 = values[8] * values[13] - values[9] * values[12];
			return values[0] * (values[5] * s0 - values[6] * s1 + values[7] * s2) -
				values[1] * (values[4] * s0 - values[6] * s3 + values[7] * s4) +
				values[2] * (values[4] * s1 - values[5] * s3 + values[7] * s5) -
				values[3] * (values[4] * s2 - values[5] * s4 + values[6] * s5);
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::invert() noexcept {
			const T det = determinant();
			if (det == static_cast<T>(0)) {
				return *this;
			}

			const T adjugate[16] = {
				values[7] * values[9] * values[14] - values[7] * values[10] * values[13] + values[6] * values[11] * values[13] - values[5] * values[11] * values[14] - values[6] * values[9] * values[15] + values[5] * values[10] * values[15],
				values[3] * values[10] * values[13] - values[3] * values[9] * values[14] - values[2] * values[11] * values[13] + values[1] * values[11] * values[14] + values[2] * values[9] * values[15] - values[1] * values[10] * values[15],
				values[3] * values[5] * values[14] - values[3] * values[6] * values[13] + values[2] * values[7] * values[13] - values[1] * values[7] * values[14] - values[2] * values[5] * values[15] + values[1] * values[6] * values[15],
				values[3] * values[6] * values[9] - values[3] * values[5] * values[10] - values[2] * values[7] * values[9] + values[1] * values[7] * values[10] + values[2] * values[5] * values[11] - values[1] * values[6] * values[11],
				values[7] * values[10] * values[12] - values[7] * values[8] * values[14] - values[6] * values[11] * values[12] + values[4] * values[11] * values[14] + values[6] * values[8] * values[15] - values[4] * values[10] * values[15],
				values[3] * values[8] * values[14] - values[3] * values[10] * values[12] + values[2] * values[11] * values[12] - values[0] * values[11] * values[14] - values[2] * values[8] * values[15] + values[0] * values[10] * values[15],
				values[3] * values[6] * values[12] - values[3] * values[4] * values[14] - values[2] * values[7] * values[12] + values[0] * values[7] * values[14] + values[2] * values[4] * values[15] - values[0] * values[6] * values[15],
				values[3] * values[4] * values[10] - values[3] * values[6] * values[8] + values[2] * values[7] * values[8] - values[0] * values[7] * values[10] - values[2] * values[4] * values[11] + values[0] * values[6] * values[11],
				values[7] * values[8] * values[13] - values[7] * values[9] * values[12] + values[5] * values[11] * values[12] - values[4] * values[11] * values[13] - values[5] * values[8] * values[15] + values[4] * values[9] * values[15],
				values[3] * values[9] * values[12] - values[3] * values[8] * values[13] - values[1] * values[11] * values[12] + values[0] * values[11] * values[13] + values[1] * values[8] * values[15] - values[0] * values[9] * values[15],
				values[3] * values[4] * values[13] - values[3] * values[5] * values[12] + values[1] * values[7] * values[12] - values[0] * values[7] * values[13] - values[1] * values[4] * values[15] + values[0] * values[5] * values[15],
				values[3] * values[5] * values[8] - values[3] * values[4] * values[9] - values[1] * values[7] * values[8] + values[0] * values[7] * values[9] + values[1] * values[4] * values[11] - values[0] * values[5] * values[11],
				values[6] * values[9] * values[12] - values[6] * values[8] * values[13] - values[5] * values[10] * values[12] + values[4] * values[10] * values[13] + values[5] * values[8] * values[14] - values[4] * values[9] * values[14],
				values[2] * values[8] * values[13] - values[2] * values[9] * values[12] + values[1] * values[10] * values[12] - values[0] * values[10] * values[13] - values[1] * values[8] * values[14] + values[0] * values[9] * values[14],
				values[2] * values[5] * values[12] - values[2] * values[4] * values[13] - values[1] * values[6] * values[12] + values[0] * values[6] * values[13] + values[1] * values[4] * values[14] - values[0] * values[5] * values[14],
				values[2] * values[4] * values[9] - values[2] * values[5] * values[8] + values[1] * values[6] * values[8] - values[0] * values[6] * values[9] - values[1] * values[4] * values[10] + values[0] * values[5] * values[10]
			};

			for (std::size_t i = 0; i < 16; i++) {
				values[i] = adjugate[i] / det;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::transpose() noexcept {
			const T transposed[16] = {
				values[0], values[4], values[8], values[12],
				values[1], values[5], values[9], values[13],
				values[2], values[6], values[10], values[14],
				values[3], values[7], values[11], values[15]
			};

			for (std::size_t i = 0; i < 16; i++) {
				values[i] = transposed[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr bool mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const mat4& matrix) const noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				if (values[i] != matrix.values[i]) {
					return false;
				}
			}
			return true;
		}

		template <typename T, std::size_t A>
		constexpr bool mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const mat4& matrix) const noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				if (values[i] != matrix.values[i]) {
					return true;
				}
			}
			return false;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const mat4& matrix) noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				values[i] += matrix.values[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const mat4& matrix) noexcept {
			for (std::size_t i = 0; i < 16; i++) {
				values[i] -= matrix.values[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(const mat4& matrix) noexcept {
			const mat4<T, A> product = *this * matrix;
			for (std::size_t i = 0; i < 16; i++) {
				values[i] = product.values[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator+(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
			mat4<T, A> result;
			for (std::size_t i = 0; i < 16; i++) {
				result.values[i] = left.values[i] + right.values[i];
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator-(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
			mat4<T, A> result;
			for (std::size_t i = 0; i < 16; i++) {
				result.values[i] = left.values[i] - right.values[i];
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator*(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
			mat4<T, A> result;
			for (std::size_t row = 0; row < 4; row++) {
				for (std::size_t column = 0; column < 4; column++) {
					result.values[row * 4 + column] =
						left.values[row * 4] * right.values[column] +
						left.values[row * 4 + 1] * right.values[4 + column] +
						left.values[row * 4 + 2] * right.values[8 + column] +
						left.values[row * 4 + 3] * right.values[12 + column];
				}
			}
			return result;
		}

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const mat4<T, A>& matrix, const vec3<T, VA>& vector) noexcept {
			return {
				matrix.values[0] * vector.x + matrix.values[1] * vector.y + matrix.values[2] * vector.z + matrix.values[3],
				matrix.values[4] * vector.x + matrix.values[5] * vector.y + matrix.values[6] * vector.z + matrix.values[7],
				matrix.values[8] * vector.x + matrix.values[9] * vector.y + matrix.values[10] * vector.z + matrix.values[11]
			};
		}

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const vec3<T, VA>& vector, const mat4<T, A>& matrix) noexcept {
			return {
				matrix.values[0] * vector.x + matrix.values[1] * vector.y + matrix.values[2] * vector.z + matrix.values[3],
				matrix.values[4] * vector.x + matrix.values[5] * vector.y + matrix.values[6] * vector.z + matrix.values[7],
				matrix.values[8] * vector.x + matrix.values[9] * vector.y + matrix.values[10] * vector.z + matrix.values[11]
			};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
