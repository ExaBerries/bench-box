#pragma once
#include <cstddef>
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat3(const T (&nvalues)[9]) noexcept {
			set(nvalues);
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat3(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8) noexcept {
			values[0] = v0;
			values[1] = v1;
			values[2] = v2;
			values[3] = 0;
			values[4] = v3;
			values[5] = v4;
			values[6] = v5;
			values[7] = 0;
			values[8] = v6;
			values[9] = v7;
			values[10] = v8;
			values[11] = 0;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::setIdentity() noexcept {
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
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const T (&nvalues)[9]) noexcept {
			for (std::size_t row = 0; row < 3; row++) {
				values[row * 4] = nvalues[row * 3];
				values[row * 4 + 1] = nvalues[row * 3 + 1];
				values[row * 4 + 2] = nvalues[row * 3 + 2];
				values[row * 4 + 3] = 0;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr T mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::determinant() const noexcept {
			const T m00 = values[0];
			const T m01 = values[1];
			const T m02 = values[2];
			const T m10 = values[4];
			const T m11 = values[5];
			const T m12 = values[6];
			const T m20 = values[8];
			const T m21 = values[9];
			const T m22 = values[10];
			return m00 * (m11 * m22 - m12 * m21) -
				m01 * (m10 * m22 - m12 * m20) +
				m02 * (m10 * m21 - m11 * m20);
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::invert() noexcept {
			const T det = determinant();
			if (det == static_cast<T>(0)) {
				return *this;
			}

			const T inv_det = static_cast<T>(1) / det;
			const T m00 = values[0];
			const T m01 = values[1];
			const T m02 = values[2];
			const T m10 = values[4];
			const T m11 = values[5];
			const T m12 = values[6];
			const T m20 = values[8];
			const T m21 = values[9];
			const T m22 = values[10];

			values[0] = (m11 * m22 - m12 * m21) * inv_det;
			values[1] = (m02 * m21 - m01 * m22) * inv_det;
			values[2] = (m01 * m12 - m02 * m11) * inv_det;
			values[3] = 0;
			values[4] = (m12 * m20 - m10 * m22) * inv_det;
			values[5] = (m00 * m22 - m02 * m20) * inv_det;
			values[6] = (m02 * m10 - m00 * m12) * inv_det;
			values[7] = 0;
			values[8] = (m10 * m21 - m11 * m20) * inv_det;
			values[9] = (m01 * m20 - m00 * m21) * inv_det;
			values[10] = (m00 * m11 - m01 * m10) * inv_det;
			values[11] = 0;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::transpose() noexcept {
			// three swaps beat any shuffle sequence here
			const T m01 = values[1];
			const T m02 = values[2];
			const T m12 = values[6];
			values[1] = values[4];
			values[4] = m01;
			values[2] = values[8];
			values[8] = m02;
			values[6] = values[9];
			values[9] = m12;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr bool mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const mat3& matrix) const noexcept {
			for (std::size_t row = 0; row < 3; row++) {
				for (std::size_t column = 0; column < 3; column++) {
					if (values[row * 4 + column] != matrix.values[row * 4 + column]) {
						return false;
					}
				}
			}
			return true;
		}

		template <typename T, std::size_t A>
		constexpr bool mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const mat3& matrix) const noexcept {
			return !(*this == matrix);
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const mat3& matrix) noexcept {
			*this = *this + matrix;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const mat3& matrix) noexcept {
			*this = *this - matrix;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat3<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(const mat3& matrix) noexcept {
			*this = *this * matrix;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator+(const mat3<T, A>& left, const mat3<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				mat3<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					static_assert(A >= alignof(simd::reg4f), "mat3 rows must be aligned for SIMD loads");
					simd::store4f(simd_result.values[0], simd::add4f(simd::load4f4f(left.values[0]), simd::load4f4f(right.values[0])));
					simd::store4f(simd_result.values[4], simd::add4f(simd::load4f4f(left.values[4]), simd::load4f4f(right.values[4])));
					simd::store4f(simd_result.values[8], simd::add4f(simd::load4f4f(left.values[8]), simd::load4f4f(right.values[8])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					static_assert(A >= alignof(simd::reg4d), "mat3 rows must be aligned for SIMD loads");
					simd::store4d(simd_result.values[0], simd::add4d(simd::load4d4d(left.values[0]), simd::load4d4d(right.values[0])));
					simd::store4d(simd_result.values[4], simd::add4d(simd::load4d4d(left.values[4]), simd::load4d4d(right.values[4])));
					simd::store4d(simd_result.values[8], simd::add4d(simd::load4d4d(left.values[8]), simd::load4d4d(right.values[8])));
					return simd_result;
				}
			}
		#endif
			mat3<T, A> result;
			for (std::size_t row = 0; row < 3; row++) {
				result.values[row * 4] = left.values[row * 4] + right.values[row * 4];
				result.values[row * 4 + 1] = left.values[row * 4 + 1] + right.values[row * 4 + 1];
				result.values[row * 4 + 2] = left.values[row * 4 + 2] + right.values[row * 4 + 2];
				result.values[row * 4 + 3] = 0;
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator-(const mat3<T, A>& left, const mat3<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				mat3<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					static_assert(A >= alignof(simd::reg4f), "mat3 rows must be aligned for SIMD loads");
					simd::store4f(simd_result.values[0], simd::sub4f(simd::load4f4f(left.values[0]), simd::load4f4f(right.values[0])));
					simd::store4f(simd_result.values[4], simd::sub4f(simd::load4f4f(left.values[4]), simd::load4f4f(right.values[4])));
					simd::store4f(simd_result.values[8], simd::sub4f(simd::load4f4f(left.values[8]), simd::load4f4f(right.values[8])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					static_assert(A >= alignof(simd::reg4d), "mat3 rows must be aligned for SIMD loads");
					simd::store4d(simd_result.values[0], simd::sub4d(simd::load4d4d(left.values[0]), simd::load4d4d(right.values[0])));
					simd::store4d(simd_result.values[4], simd::sub4d(simd::load4d4d(left.values[4]), simd::load4d4d(right.values[4])));
					simd::store4d(simd_result.values[8], simd::sub4d(simd::load4d4d(left.values[8]), simd::load4d4d(right.values[8])));
					return simd_result;
				}
			}
		#endif
			mat3<T, A> result;
			for (std::size_t row = 0; row < 3; row++) {
				result.values[row * 4] = left.values[row * 4] - right.values[row * 4];
				result.values[row * 4 + 1] = left.values[row * 4 + 1] - right.values[row * 4 + 1];
				result.values[row * 4 + 2] = left.values[row * 4 + 2] - right.values[row * 4 + 2];
				result.values[row * 4 + 3] = 0;
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat3<T, A> operator*(const mat3<T, A>& left, const mat3<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					static_assert(A >= alignof(simd::reg4f), "mat3 rows must be aligned for SIMD loads");
					// result row r = left[r][0] * right_row_0 + left[r][1] * right_row_1 + left[r][2] * right_row_2
					const simd::reg4f r0 = simd::load4f4f(right.values[0]);
					const simd::reg4f r1 = simd::load4f4f(right.values[4]);
					const simd::reg4f r2 = simd::load4f4f(right.values[8]);
					mat3<T, A> simd_result;
					simd::reg4f acc = simd::mul4f(r0, simd::load1f4f(left.values[0]));
					acc = simd::fmadd4f(r1, simd::load1f4f(left.values[1]), acc);
					acc = simd::fmadd4f(r2, simd::load1f4f(left.values[2]), acc);
					simd::store4f(simd_result.values[0], acc);
					acc = simd::mul4f(r0, simd::load1f4f(left.values[4]));
					acc = simd::fmadd4f(r1, simd::load1f4f(left.values[5]), acc);
					acc = simd::fmadd4f(r2, simd::load1f4f(left.values[6]), acc);
					simd::store4f(simd_result.values[4], acc);
					acc = simd::mul4f(r0, simd::load1f4f(left.values[8]));
					acc = simd::fmadd4f(r1, simd::load1f4f(left.values[9]), acc);
					acc = simd::fmadd4f(r2, simd::load1f4f(left.values[10]), acc);
					simd::store4f(simd_result.values[8], acc);
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					static_assert(A >= alignof(simd::reg4d), "mat3 rows must be aligned for SIMD loads");
					const simd::reg4d r0 = simd::load4d4d(right.values[0]);
					const simd::reg4d r1 = simd::load4d4d(right.values[4]);
					const simd::reg4d r2 = simd::load4d4d(right.values[8]);
					mat3<T, A> simd_result;
					simd::reg4d acc = simd::mul4d(r0, simd::load1d4d(left.values[0]));
					acc = simd::fmadd4d(r1, simd::load1d4d(left.values[1]), acc);
					acc = simd::fmadd4d(r2, simd::load1d4d(left.values[2]), acc);
					simd::store4d(simd_result.values[0], acc);
					acc = simd::mul4d(r0, simd::load1d4d(left.values[4]));
					acc = simd::fmadd4d(r1, simd::load1d4d(left.values[5]), acc);
					acc = simd::fmadd4d(r2, simd::load1d4d(left.values[6]), acc);
					simd::store4d(simd_result.values[4], acc);
					acc = simd::mul4d(r0, simd::load1d4d(left.values[8]));
					acc = simd::fmadd4d(r1, simd::load1d4d(left.values[9]), acc);
					acc = simd::fmadd4d(r2, simd::load1d4d(left.values[10]), acc);
					simd::store4d(simd_result.values[8], acc);
					return simd_result;
				}
			}
		#endif
			mat3<T, A> result;
			for (std::size_t row = 0; row < 3; row++) {
				for (std::size_t column = 0; column < 3; column++) {
					result.values[row * 4 + column] =
						left.values[row * 4] * right.values[column] +
						left.values[row * 4 + 1] * right.values[4 + column] +
						left.values[row * 4 + 2] * right.values[8 + column];
				}
				result.values[row * 4 + 3] = 0;
			}
			return result;
		}

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const mat3<T, A>& matrix, const vec3<T, VA>& vector) noexcept {
			// first attempt and manual simd was slower so leaving it to the compiler for now
			return {
				matrix.values[0] * vector.x + matrix.values[1] * vector.y + matrix.values[2] * vector.z,
				matrix.values[4] * vector.x + matrix.values[5] * vector.y + matrix.values[6] * vector.z,
				matrix.values[8] * vector.x + matrix.values[9] * vector.y + matrix.values[10] * vector.z
			};
		}

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const vec3<T, VA>& vector, const mat3<T, A>& matrix) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4f), "vec3 must be aligned for SIMD loads");
					static_assert(sizeof(vec3<T, VA>) >= 4 * sizeof(T), "vec3 must have a full padding lane for SIMD stores");
					const simd::reg4f res = simd::fmadd4f(
						simd::load4f4f(matrix.values[8]), simd::load1f4f(vector.z),
						simd::fmadd4f(
							simd::load4f4f(matrix.values[4]), simd::load1f4f(vector.y),
							simd::mul4f(simd::load4f4f(matrix.values[0]), simd::load1f4f(vector.x))
						)
					);
					vec3<T, VA> out;
					simd::store4f(out.x, res);
					return out;
				} else if constexpr (std::is_same_v<T, double>) {
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4d), "vec3 must be aligned for SIMD loads");
					static_assert(sizeof(vec3<T, VA>) >= 4 * sizeof(T), "vec3 must have a full padding lane for SIMD stores");
					const simd::reg4d res = simd::fmadd4d(
						simd::load4d4d(matrix.values[8]), simd::load1d4d(vector.z),
						simd::fmadd4d(
							simd::load4d4d(matrix.values[4]), simd::load1d4d(vector.y),
							simd::mul4d(simd::load4d4d(matrix.values[0]), simd::load1d4d(vector.x))
						)
					);
					vec3<T, VA> out;
					simd::store4d(out.x, res);
					return out;
				}
			}
		#endif
			return {
				matrix.values[0] * vector.x + matrix.values[4] * vector.y + matrix.values[8] * vector.z,
				matrix.values[1] * vector.x + matrix.values[5] * vector.y + matrix.values[9] * vector.z,
				matrix.values[2] * vector.x + matrix.values[6] * vector.y + matrix.values[10] * vector.z
			};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
