#pragma once
#include <cstddef>
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::mat4(const T (&nvalues)[16]) noexcept {
			for (auto i = 0u; i < 16u; i++) {
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
			for (auto i = 0u; i < 16u; i++) {
				values[i] = nvalues[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		template <std::size_t VA, std::size_t QA>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::set(const vec3<T, VA>& translation, const quat<T, QA>& rotation, const vec3<T, VA>& scale) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				// float only because we become overly permute bound with reg8d compared to reg4d
				if constexpr (std::is_same_v<T, float>) {
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4f), "vec3 must be aligned for SIMD loads");
					static_assert(alignof(quat<T, QA>) >= alignof(simd::reg4f), "quat must be aligned for SIMD loads");
					const simd::reg16f zero = simd::zero_reg16f();
					const simd::reg16f one = simd::load1f16f(static_cast<T>(1));
					const simd::reg16f qw = simd::load1f16f(rotation.w);

					// quat::x..w are contiguous, so one 16-byte broadcast gives q = [x, y, z, w] repeated in each row
					const simd::reg16f q = simd::broadcast4f16f(simd::load4f4f(rotation.x));

					// 2 * [x*x, x*y, x*z, 0, y*x, y*y, y*z, 0, z*x, z*y, z*z, 0, 0, 0, 0, 0]
					const simd::reg16f qr = simd::permutex2var16f(q, zero, simd::create_mask_16i32<0, 0, 0, 16, 1, 1, 1, 16, 2, 2, 2, 16, 16, 16, 16, 16>());
					const simd::reg16f qc = simd::permutexvar16f(q, simd::create_mask_16i32<0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0>());
					const simd::reg16f p = simd::mul16f(qr, qc);
					const simd::reg16f p2 = simd::add16f(p, p);

					// [0, 0, y, 0, z, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0], then the skew part
					const simd::reg16f c = simd::permutex2var16f(zero, q, simd::create_mask_16i32<0, 0, 17, 0, 18, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0>());
					const simd::reg16f wc = simd::mul16f(qw, c);
					const simd::reg16f wt = simd::permutexvar16f(wc, simd::create_mask_16i32<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>());
					const simd::reg16f skew = simd::sub16f(wc, wt);
					const simd::reg16f skew2 = simd::add16f(skew, skew);

					// Diagonal: compute 1 - 2*(x*x + y*y + z*z) with scalar FMA, then merge it into lanes 0, 5 and 10 of the rotation matrix
					const T xx = rotation.x * rotation.x;
					const T yy = rotation.y * rotation.y;
					const T zz = rotation.z * rotation.z;
					const simd::reg16f cc = simd::load1f16f(static_cast<T>(1) - static_cast<T>(2) * (xx + yy + zz));
					const simd::reg16f base = simd::add16f(p2, skew2);
					const simd::reg16f rot = simd::fmadd16fmask3(cc, one, base, static_cast<simd::mask16>(0x0421));

					// [scale.x, scale.y, scale.z, 1] and [tx, ty, tz, 1] built in 128 bits and broadcast, avoiding several 512-bit permutes.
					const simd::reg4f s4 = simd::blend4f<0b1000>(
						simd::load4f4f(scale.x),
						simd::load1f4f(static_cast<T>(1))
					);
					const simd::reg16f s = simd::broadcast4f16f(s4);

					const simd::reg4f t4s = simd::blend4f<0b1000>(
						simd::load4f4f(translation.x),
						simd::load1f4f(static_cast<T>(1))
					);
					const simd::reg16f tb = simd::broadcast4f16f(t4s);
					const simd::reg16f trans = simd::permutex2var16f(zero, tb, simd::create_mask_16i32<0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 19>());

					simd::store16f(values[0], simd::fmadd16f(rot, s, trans));
					return *this;
				}
			}
		#endif
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
				#if !defined(BENCH_BOX_SIMD_USE_AVX512VL)
					const simd::reg4f one = simd::load1f4f(static_cast<T>(1));
					const simd::reg4f two = simd::load1f4f(static_cast<T>(2));
					const simd::reg4f qx = simd::load1f4f(rotation.x);
					const simd::reg4f qy = simd::load1f4f(rotation.y);
					const simd::reg4f qz = simd::load1f4f(rotation.z);
					const simd::reg4f qw = simd::load1f4f(rotation.w);
					const simd::reg4f xx = simd::mul4f(qx, qx);
					const simd::reg4f yy = simd::mul4f(qy, qy);
					const simd::reg4f zz = simd::mul4f(qz, qz);
					const simd::reg4f x2 = simd::add4f(qx, qx);
					const simd::reg4f y2 = simd::add4f(qy, qy);
					const simd::reg4f w2 = simd::add4f(qw, qw);
					const simd::reg4f xy2 = simd::mul4f(x2, qy);
					const simd::reg4f xz2 = simd::mul4f(x2, qz);
					const simd::reg4f yz2 = simd::mul4f(y2, qz);
					const simd::reg4f wx2 = simd::mul4f(w2, qx);
					const simd::reg4f wy2 = simd::mul4f(w2, qy);
					const simd::reg4f wz2 = simd::mul4f(w2, qz);
					const simd::reg4f d0 = simd::fmnadd4f(two, simd::add4f(yy, zz), one);
					const simd::reg4f d1 = simd::fmnadd4f(two, simd::add4f(xx, zz), one);
					const simd::reg4f d2 = simd::fmnadd4f(two, simd::add4f(xx, yy), one);
					const simd::reg4f v0 = simd::blend4f<0b1000>(simd::blend4f<0b0100>(simd::blend4f<0b0010>(d0, simd::sub4f(xy2, wz2)), simd::add4f(xz2, wy2)), simd::zero_reg4f());
					const simd::reg4f v1 = simd::blend4f<0b1000>(simd::blend4f<0b0100>(simd::blend4f<0b0010>(simd::add4f(xy2, wz2), d1), simd::sub4f(yz2, wx2)), simd::zero_reg4f());
					const simd::reg4f v2 = simd::blend4f<0b1000>(simd::blend4f<0b0100>(simd::blend4f<0b0010>(simd::sub4f(xz2, wy2), simd::add4f(yz2, wx2)), d2), simd::zero_reg4f());
					const simd::reg4f vs = simd::blend4f<0b1000>(simd::blend4f<0b0100>(simd::blend4f<0b0010>(simd::load1f4f(scale.x), simd::load1f4f(scale.y)), simd::load1f4f(scale.z)), one);
					const simd::reg4f t0 = simd::blend4f<0b1000>(simd::zero_reg4f(), simd::load1f4f(translation.x));
					const simd::reg4f t1 = simd::blend4f<0b1000>(simd::zero_reg4f(), simd::load1f4f(translation.y));
					const simd::reg4f t2 = simd::blend4f<0b1000>(simd::zero_reg4f(), simd::load1f4f(translation.z));
					simd::store4f(values[0], simd::fmadd4f(v0, vs, t0));
					simd::store4f(values[4], simd::fmadd4f(v1, vs, t1));
					simd::store4f(values[8], simd::fmadd4f(v2, vs, t2));
					simd::store4f(values[12], simd::blend4f<0b1000>(simd::zero_reg4f(), one));
					return *this;
				#endif
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg4d one = simd::load1d4d(static_cast<T>(1));
					const simd::reg4d two = simd::load1d4d(static_cast<T>(2));
					const simd::reg4d qx = simd::load1d4d(rotation.x);
					const simd::reg4d qy = simd::load1d4d(rotation.y);
					const simd::reg4d qz = simd::load1d4d(rotation.z);
					const simd::reg4d qw = simd::load1d4d(rotation.w);
					const simd::reg4d xx = simd::mul4d(qx, qx);
					const simd::reg4d yy = simd::mul4d(qy, qy);
					const simd::reg4d zz = simd::mul4d(qz, qz);
					const simd::reg4d x2 = simd::add4d(qx, qx);
					const simd::reg4d y2 = simd::add4d(qy, qy);
					const simd::reg4d w2 = simd::add4d(qw, qw);
					const simd::reg4d xy2 = simd::mul4d(x2, qy);
					const simd::reg4d xz2 = simd::mul4d(x2, qz);
					const simd::reg4d yz2 = simd::mul4d(y2, qz);
					const simd::reg4d wx2 = simd::mul4d(w2, qx);
					const simd::reg4d wy2 = simd::mul4d(w2, qy);
					const simd::reg4d wz2 = simd::mul4d(w2, qz);
					const simd::reg4d d0 = simd::fmnadd4d(two, simd::add4d(yy, zz), one);
					const simd::reg4d d1 = simd::fmnadd4d(two, simd::add4d(xx, zz), one);
					const simd::reg4d d2 = simd::fmnadd4d(two, simd::add4d(xx, yy), one);
					const simd::reg4d v0 = simd::blend4d<0b1000>(simd::blend4d<0b0100>(simd::blend4d<0b0010>(d0, simd::sub4d(xy2, wz2)), simd::add4d(xz2, wy2)), simd::zero_reg4d());
					const simd::reg4d v1 = simd::blend4d<0b1000>(simd::blend4d<0b0100>(simd::blend4d<0b0010>(simd::add4d(xy2, wz2), d1), simd::sub4d(yz2, wx2)), simd::zero_reg4d());
					const simd::reg4d v2 = simd::blend4d<0b1000>(simd::blend4d<0b0100>(simd::blend4d<0b0010>(simd::sub4d(xz2, wy2), simd::add4d(yz2, wx2)), d2), simd::zero_reg4d());
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4d), "vec3 must be aligned for SIMD loads");
					const simd::reg4d vs = simd::blend4d<0b1000>(simd::load4d4d(scale.x), one);
					const simd::reg4d t0 = simd::blend4d<0b1000>(simd::zero_reg4d(), simd::load1d4d(translation.x));
					const simd::reg4d t1 = simd::blend4d<0b1000>(simd::zero_reg4d(), simd::load1d4d(translation.y));
					const simd::reg4d t2 = simd::blend4d<0b1000>(simd::zero_reg4d(), simd::load1d4d(translation.z));
					simd::store4d(values[0], simd::fmadd4d(v0, vs, t0));
					simd::store4d(values[4], simd::fmadd4d(v1, vs, t1));
					simd::store4d(values[8], simd::fmadd4d(v2, vs, t2));
					simd::store4d(values[12], simd::blend4d<0b1000>(simd::zero_reg4d(), one));
					return *this;
				}
			}
		#endif
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
		// float prefers the AVX2 blend/permute sequence; double uses AVX512VL permutes when they are available
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					const simd::reg4f r0 = simd::load4f4f(values[0]);
					const simd::reg4f r1 = simd::load4f4f(values[4]);
					const simd::reg4f r2 = simd::load4f4f(values[8]);
					const simd::reg4f r3 = simd::load4f4f(values[12]);
					const simd::reg4f a = simd::blend4f<0b1100>(r0, simd::permute4f<0, 1, 0, 1>(r1));
					const simd::reg4f b = simd::blend4f<0b1100>(simd::permute4f<2, 3, 2, 3>(r0), simd::permute4f<2, 3, 2, 3>(r1));
					const simd::reg4f c = simd::blend4f<0b1100>(r2, simd::permute4f<0, 1, 0, 1>(r3));
					const simd::reg4f d = simd::blend4f<0b1100>(simd::permute4f<2, 3, 2, 3>(r2), simd::permute4f<2, 3, 2, 3>(r3));
					const simd::reg4f q0 = simd::permute4f<0, 2, 0, 0>(r0);
					const simd::reg4f q1 = simd::permute4f<0, 0, 0, 2>(r2);
					const simd::reg4f q2 = simd::permute4f<1, 3, 0, 0>(r1);
					const simd::reg4f q3 = simd::permute4f<0, 0, 1, 3>(r3);
					const simd::reg4f q4 = simd::permute4f<1, 3, 0, 0>(r0);
					const simd::reg4f q5 = simd::permute4f<0, 0, 1, 3>(r2);
					const simd::reg4f q6 = simd::permute4f<0, 2, 0, 0>(r1);
					const simd::reg4f q7 = simd::permute4f<0, 0, 0, 2>(r3);
					const simd::reg4f detsub = simd::fmsub4f(simd::blend4f<0b1100>(q0, q1), simd::blend4f<0b1100>(q2, q3), simd::mul4f(simd::blend4f<0b1100>(q4, q5), simd::blend4f<0b1100>(q6, q7)));
					const simd::reg4f ab = simd::fmsub4f(simd::permute4f<3, 3, 0, 0>(a), b, simd::mul4f(simd::permute4f<1, 1, 2, 2>(a), simd::permute4f<2, 3, 0, 1>(b)));
					const simd::reg4f dc = simd::fmsub4f(simd::permute4f<3, 3, 0, 0>(d), c, simd::mul4f(simd::permute4f<1, 1, 2, 2>(d), simd::permute4f<2, 3, 0, 1>(c)));
					const simd::reg4f dets_pairs = simd::mul4f(detsub, simd::permute4f<3, 2, 1, 0>(detsub));
					const simd::reg4f dets = simd::add4f(dets_pairs, simd::permute4f<1, 0, 3, 2>(dets_pairs));
					const simd::reg4f trm = simd::mul4f(ab, simd::permute4f<0, 2, 1, 3>(dc));
					const simd::reg4f trs = simd::add4f(trm, simd::permute4f<1, 0, 3, 2>(trm));
					const simd::reg4f det = simd::sub4f(dets, simd::add4f(trs, simd::permute4f<2, 3, 0, 1>(trs)));
					return simd::extract4f_lo(det);
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg4d r0 = simd::load4d4d(values[0]);
					const simd::reg4d r1 = simd::load4d4d(values[4]);
					const simd::reg4d r2 = simd::load4d4d(values[8]);
					const simd::reg4d r3 = simd::load4d4d(values[12]);
				#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
					const simd::reg4d a = simd::permutex2var4d(r0, r1, simd::create_mask_4i64<0, 1, 4, 5>());
					const simd::reg4d b = simd::permutex2var4d(r0, r1, simd::create_mask_4i64<2, 3, 6, 7>());
					const simd::reg4d c = simd::permutex2var4d(r2, r3, simd::create_mask_4i64<0, 1, 4, 5>());
					const simd::reg4d d = simd::permutex2var4d(r2, r3, simd::create_mask_4i64<2, 3, 6, 7>());
					const simd::reg4d tA = simd::permutex2var4d(r0, r2, simd::create_mask_4i64<0, 2, 4, 6>());
					const simd::reg4d tB = simd::permutex2var4d(r1, r3, simd::create_mask_4i64<1, 3, 5, 7>());
					const simd::reg4d tC = simd::permutex2var4d(r0, r2, simd::create_mask_4i64<1, 3, 5, 7>());
					const simd::reg4d tD = simd::permutex2var4d(r1, r3, simd::create_mask_4i64<0, 2, 4, 6>());
					const simd::reg4d detsub = simd::fmsub4d(tA, tB, simd::mul4d(tC, tD));
				#else
					const simd::reg4d a = simd::blend4d<0b1100>(r0, simd::permute4d<0, 1, 0, 1>(r1));
					const simd::reg4d b = simd::blend4d<0b1100>(simd::permute4d<2, 3, 2, 3>(r0), simd::permute4d<2, 3, 2, 3>(r1));
					const simd::reg4d c = simd::blend4d<0b1100>(r2, simd::permute4d<0, 1, 0, 1>(r3));
					const simd::reg4d d = simd::blend4d<0b1100>(simd::permute4d<2, 3, 2, 3>(r2), simd::permute4d<2, 3, 2, 3>(r3));
					const simd::reg4d q0 = simd::permute4d<0, 2, 0, 0>(r0);
					const simd::reg4d q1 = simd::permute4d<0, 0, 0, 2>(r2);
					const simd::reg4d q2 = simd::permute4d<1, 3, 0, 0>(r1);
					const simd::reg4d q3 = simd::permute4d<0, 0, 1, 3>(r3);
					const simd::reg4d q4 = simd::permute4d<1, 3, 0, 0>(r0);
					const simd::reg4d q5 = simd::permute4d<0, 0, 1, 3>(r2);
					const simd::reg4d q6 = simd::permute4d<0, 2, 0, 0>(r1);
					const simd::reg4d q7 = simd::permute4d<0, 0, 0, 2>(r3);
					const simd::reg4d detsub = simd::fmsub4d(simd::blend4d<0b1100>(q0, q1), simd::blend4d<0b1100>(q2, q3), simd::mul4d(simd::blend4d<0b1100>(q4, q5), simd::blend4d<0b1100>(q6, q7)));
				#endif
					const simd::reg4d ab = simd::fmsub4d(simd::permute4d<3, 3, 0, 0>(a), b, simd::mul4d(simd::permute4d<1, 1, 2, 2>(a), simd::permute4d<2, 3, 0, 1>(b)));
					const simd::reg4d dc = simd::fmsub4d(simd::permute4d<3, 3, 0, 0>(d), c, simd::mul4d(simd::permute4d<1, 1, 2, 2>(d), simd::permute4d<2, 3, 0, 1>(c)));
					const simd::reg4d dets = simd::fmadd4d(simd::permute4d<0, 0, 0, 0>(detsub), simd::permute4d<3, 3, 3, 3>(detsub), simd::mul4d(simd::permute4d<1, 1, 1, 1>(detsub), simd::permute4d<2, 2, 2, 2>(detsub)));
					const simd::reg4d trm = simd::mul4d(ab, simd::permute4d<0, 2, 1, 3>(dc));
					const simd::reg4d trs = simd::add4d(trm, simd::permute4d<1, 0, 3, 2>(trm));
					const simd::reg4d det = simd::sub4d(dets, simd::add4d(trs, simd::permute4d<2, 3, 0, 1>(trs)));
					return simd::extract4d_lo(det);
				}
			}
		#endif
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
			#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
				if (!std::is_constant_evaluated()) {
					if constexpr (std::is_same_v<T, double>) { // only double sees an actual benefit from avx512vl
						// The 4-wide inverse is shuffle-port bound on double. Packing two adjugate vectors per register cuts the permutation count nearly in half, while float stays 128/256-bit (FMA bound).
						const simd::reg8d m01 = simd::load8d8d(values[0]);
						const simd::reg8d m23 = simd::load8d8d(values[8]);
						const simd::reg8d g1 = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<0, 2, 8, 10, 0, 2, 8, 10>());
						const simd::reg8d g2 = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<5, 7, 13, 15, 5, 7, 13, 15>());
						const simd::reg8d g3 = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<1, 3, 9, 11, 1, 3, 9, 11>());
						const simd::reg8d g4 = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<4, 6, 12, 14, 4, 6, 12, 14>());
						const simd::reg8d detsub = simd::fmsub8d(g1, g2, simd::mul8d(g3, g4));
						// Two 2x2 blocks per register: [a|d] and [b|c]
						const simd::reg8d tAD = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<0, 1, 4, 5, 10, 11, 14, 15>());
						const simd::reg8d tBC = simd::permutex2var8d(m01, m23, simd::create_mask_8i64<2, 3, 6, 7, 8, 9, 12, 13>());
						// [ab|dc] computed at 512-bit width
						const simd::reg8d tAbDc = simd::fmsub8d(
							simd::permutexvar8d(tAD, simd::create_mask_8i64<3, 3, 0, 0, 7, 7, 4, 4>()),
							tBC,
							simd::mul8d(
								simd::permutexvar8d(tAD, simd::create_mask_8i64<1, 1, 2, 2, 5, 5, 6, 6>()),
								simd::permutexvar8d(tBC, simd::create_mask_8i64<2, 3, 0, 1, 6, 7, 4, 5>())
							)
						);
						// Swapped halves are shared by both adjugate pairs
						const simd::reg8d tS2 = simd::permutexvar8d(tAD, simd::create_mask_8i64<4, 5, 6, 7, 0, 1, 2, 3>());
						const simd::reg8d tL2 = simd::permutexvar8d(tBC, simd::create_mask_8i64<4, 5, 6, 7, 0, 1, 2, 3>());
						// [wadj|xadj] in one register
						const simd::reg8d tD1 = simd::permutexvar8d(detsub, simd::create_mask_8i64<0, 0, 0, 0, 3, 3, 3, 3>());
						const simd::reg8d tXW = simd::fmsub8d(
							tD1,
							tS2,
							simd::fmadd8d(
								tL2,
								simd::permutexvar8d(tAbDc, simd::create_mask_8i64<0, 3, 0, 3, 4, 7, 4, 7>()),
								simd::mul8d(
									simd::permutexvar8d(tL2, simd::create_mask_8i64<1, 0, 3, 2, 5, 4, 7, 6>()),
									simd::permutexvar8d(tAbDc, simd::create_mask_8i64<2, 1, 2, 1, 6, 5, 6, 5>())
								)
							)
						);
						// [yadj|zadj] in one register; the second product is subtracted.
						const simd::reg8d tCn = simd::xor_bits8d(tAbDc, simd::sign8d());
						const simd::reg8d tD2 = simd::permutexvar8d(detsub, simd::create_mask_8i64<1, 1, 1, 1, 2, 2, 2, 2>());
						const simd::reg8d tYZ = simd::fmsub8d(
							tD2,
							tL2,
							simd::fmadd8d(
								tS2,
								simd::permutexvar8d(tAbDc, simd::create_mask_8i64<3, 0, 3, 0, 7, 4, 7, 4>()),
								simd::mul8d(
									simd::permutexvar8d(tS2, simd::create_mask_8i64<1, 0, 3, 2, 5, 4, 7, 6>()),
									simd::permutexvar8d(tCn, simd::create_mask_8i64<2, 1, 2, 1, 6, 5, 6, 5>())
								)
							)
						);
						// Determinant from the 2x2 blocks, 256-bit.
						const simd::reg4d detsub4 = simd::reg8d4d(detsub);
						const simd::reg4d ab4 = simd::reg8d4d(tAbDc);
						const simd::reg4d dc4 = simd::reg8d4d(simd::permutexvar8d(tAbDc, simd::create_mask_8i64<4, 5, 6, 7, 4, 5, 6, 7>()));
						const simd::reg4d dets_pairs = simd::mul4d(detsub4, simd::permute4d<3, 2, 1, 0>(detsub4));
						const simd::reg4d dets = simd::add4d(dets_pairs, simd::permute4d<1, 0, 3, 2>(dets_pairs));
						const simd::reg4d trm = simd::mul4d(ab4, simd::permute4d<0, 2, 1, 3>(dc4));
						const simd::reg4d trs = simd::add4d(trm, simd::permute4d<1, 0, 3, 2>(trm));
						const simd::reg4d detm = simd::sub4d(dets, simd::add4d(trs, simd::permute4d<2, 3, 0, 1>(trs)));
						if (simd::extract4d_lo(detm) == static_cast<T>(0)) {
							return *this;
						}
						const simd::reg4d rdet = simd::div4d(simd::blend4d<0b0110>(simd::load1d4d(static_cast<T>(1)), simd::sub4d(simd::zero_reg4d(), simd::load1d4d(static_cast<T>(1)))), detm);
						const simd::reg8d rdet8 = simd::broadcast4d8d(rdet);
						const simd::reg8d tXWs = simd::mul8d(tXW, rdet8);
						const simd::reg8d tYZs = simd::mul8d(tYZ, rdet8);
						simd::store8d(values[0], simd::permutex2var8d(tXWs, tYZs, simd::create_mask_8i64<7, 5, 11, 9, 6, 4, 10, 8>()));
						simd::store8d(values[8], simd::permutex2var8d(tXWs, tYZs, simd::create_mask_8i64<15, 13, 3, 1, 14, 12, 2, 0>()));
						return *this;
					}
				}
			#endif
			#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
				if (!std::is_constant_evaluated()) {
					if constexpr (std::is_same_v<T, float>) {
						const simd::reg4f r0 = simd::load4f4f(values[0]);
						const simd::reg4f r1 = simd::load4f4f(values[4]);
						const simd::reg4f r2 = simd::load4f4f(values[8]);
						const simd::reg4f r3 = simd::load4f4f(values[12]);
						const simd::reg4f a = simd::blend4f<0b1100>(r0, simd::permute4f<0, 1, 0, 1>(r1));
						const simd::reg4f b = simd::blend4f<0b1100>(simd::permute4f<2, 3, 2, 3>(r0), simd::permute4f<2, 3, 2, 3>(r1));
						const simd::reg4f c = simd::blend4f<0b1100>(r2, simd::permute4f<0, 1, 0, 1>(r3));
						const simd::reg4f d = simd::blend4f<0b1100>(simd::permute4f<2, 3, 2, 3>(r2), simd::permute4f<2, 3, 2, 3>(r3));
						const simd::reg4f q0 = simd::permute4f<0, 2, 0, 0>(r0);
						const simd::reg4f q1 = simd::permute4f<0, 0, 0, 2>(r2);
						const simd::reg4f q2 = simd::permute4f<1, 3, 0, 0>(r1);
						const simd::reg4f q3 = simd::permute4f<0, 0, 1, 3>(r3);
						const simd::reg4f q4 = simd::permute4f<1, 3, 0, 0>(r0);
						const simd::reg4f q5 = simd::permute4f<0, 0, 1, 3>(r2);
						const simd::reg4f q6 = simd::permute4f<0, 2, 0, 0>(r1);
						const simd::reg4f q7 = simd::permute4f<0, 0, 0, 2>(r3);
						const simd::reg4f detsub = simd::fmsub4f(simd::blend4f<0b1100>(q0, q1), simd::blend4f<0b1100>(q2, q3), simd::mul4f(simd::blend4f<0b1100>(q4, q5), simd::blend4f<0b1100>(q6, q7)));
						const simd::reg4f ds0 = simd::permute4f<0, 0, 0, 0>(detsub);
						const simd::reg4f ds1 = simd::permute4f<1, 1, 1, 1>(detsub);
						const simd::reg4f ds2 = simd::permute4f<2, 2, 2, 2>(detsub);
						const simd::reg4f ds3 = simd::permute4f<3, 3, 3, 3>(detsub);
						const simd::reg4f ab = simd::fmsub4f(simd::permute4f<3, 3, 0, 0>(a), b, simd::mul4f(simd::permute4f<1, 1, 2, 2>(a), simd::permute4f<2, 3, 0, 1>(b)));
						const simd::reg4f dc = simd::fmsub4f(simd::permute4f<3, 3, 0, 0>(d), c, simd::mul4f(simd::permute4f<1, 1, 2, 2>(d), simd::permute4f<2, 3, 0, 1>(c)));
						const simd::reg4f xadj = simd::fmsub4f(ds3, a, simd::fmadd4f(b, simd::permute4f<0, 3, 0, 3>(dc), simd::mul4f(simd::permute4f<1, 0, 3, 2>(b), simd::permute4f<2, 1, 2, 1>(dc))));
						const simd::reg4f wadj = simd::fmsub4f(ds0, d, simd::fmadd4f(c, simd::permute4f<0, 3, 0, 3>(ab), simd::mul4f(simd::permute4f<1, 0, 3, 2>(c), simd::permute4f<2, 1, 2, 1>(ab))));
						const simd::reg4f yadj = simd::fmsub4f(ds1, c, simd::fmsub4f(d, simd::permute4f<3, 0, 3, 0>(ab), simd::mul4f(simd::permute4f<1, 0, 3, 2>(d), simd::permute4f<2, 1, 2, 1>(ab))));
						const simd::reg4f zadj = simd::fmsub4f(ds2, b, simd::fmsub4f(a, simd::permute4f<3, 0, 3, 0>(dc), simd::mul4f(simd::permute4f<1, 0, 3, 2>(a), simd::permute4f<2, 1, 2, 1>(dc))));
						const simd::reg4f trm = simd::mul4f(ab, simd::permute4f<0, 2, 1, 3>(dc));
						const simd::reg4f trs = simd::add4f(trm, simd::permute4f<1, 0, 3, 2>(trm));
						const simd::reg4f detm = simd::sub4f(simd::fmadd4f(ds0, ds3, simd::mul4f(ds1, ds2)), simd::add4f(trs, simd::permute4f<2, 3, 0, 1>(trs)));
						if (simd::extract4f_lo(detm) == static_cast<T>(0)) {
							return *this;
						}
						const simd::reg4f rdet = simd::div4f(simd::blend4f<0b0110>(simd::load1f4f(static_cast<T>(1)), simd::sub4f(simd::zero_reg4f(), simd::load1f4f(static_cast<T>(1)))), detm);
						const simd::reg4f xs = simd::mul4f(xadj, rdet);
						const simd::reg4f ys = simd::mul4f(yadj, rdet);
						const simd::reg4f zs = simd::mul4f(zadj, rdet);
						const simd::reg4f ws = simd::mul4f(wadj, rdet);
							simd::store4f(values[0], simd::blend4f<0b1100>(simd::permute4f<3, 1, 0, 0>(xs), simd::permute4f<0, 0, 3, 1>(ys)));
						simd::store4f(values[4], simd::blend4f<0b1100>(simd::permute4f<2, 0, 0, 0>(xs), simd::permute4f<0, 0, 2, 0>(ys)));
						simd::store4f(values[8], simd::blend4f<0b1100>(simd::permute4f<3, 1, 0, 0>(zs), simd::permute4f<0, 0, 3, 1>(ws)));
						simd::store4f(values[12], simd::blend4f<0b1100>(simd::permute4f<2, 0, 0, 0>(zs), simd::permute4f<0, 0, 2, 0>(ws)));
							return *this;
					} else if constexpr (std::is_same_v<T, double>) {
					#if !defined(BENCH_BOX_SIMD_USE_AVX512VL) // if we have avx512vl that path above is faster
						const simd::reg4d r0 = simd::load4d4d(values[0]);
						const simd::reg4d r1 = simd::load4d4d(values[4]);
						const simd::reg4d r2 = simd::load4d4d(values[8]);
						const simd::reg4d r3 = simd::load4d4d(values[12]);
						const simd::reg4d a = simd::blend4d<0b1100>(r0, simd::permute4d<0, 1, 0, 1>(r1));
						const simd::reg4d b = simd::blend4d<0b1100>(simd::permute4d<2, 3, 2, 3>(r0), simd::permute4d<2, 3, 2, 3>(r1));
						const simd::reg4d c = simd::blend4d<0b1100>(r2, simd::permute4d<0, 1, 0, 1>(r3));
						const simd::reg4d d = simd::blend4d<0b1100>(simd::permute4d<2, 3, 2, 3>(r2), simd::permute4d<2, 3, 2, 3>(r3));
						const simd::reg4d q0 = simd::permute4d<0, 2, 0, 0>(r0);
						const simd::reg4d q1 = simd::permute4d<0, 0, 0, 2>(r2);
						const simd::reg4d q2 = simd::permute4d<1, 3, 0, 0>(r1);
						const simd::reg4d q3 = simd::permute4d<0, 0, 1, 3>(r3);
						const simd::reg4d q4 = simd::permute4d<1, 3, 0, 0>(r0);
						const simd::reg4d q5 = simd::permute4d<0, 0, 1, 3>(r2);
						const simd::reg4d q6 = simd::permute4d<0, 2, 0, 0>(r1);
						const simd::reg4d q7 = simd::permute4d<0, 0, 0, 2>(r3);
						const simd::reg4d detsub = simd::fmsub4d(simd::blend4d<0b1100>(q0, q1), simd::blend4d<0b1100>(q2, q3), simd::mul4d(simd::blend4d<0b1100>(q4, q5), simd::blend4d<0b1100>(q6, q7)));
						const simd::reg4d ds0 = simd::permute4d<0, 0, 0, 0>(detsub);
						const simd::reg4d ds1 = simd::permute4d<1, 1, 1, 1>(detsub);
						const simd::reg4d ds2 = simd::permute4d<2, 2, 2, 2>(detsub);
						const simd::reg4d ds3 = simd::permute4d<3, 3, 3, 3>(detsub);
						const simd::reg4d ab = simd::fmsub4d(simd::permute4d<3, 3, 0, 0>(a), b, simd::mul4d(simd::permute4d<1, 1, 2, 2>(a), simd::permute4d<2, 3, 0, 1>(b)));
						const simd::reg4d dc = simd::fmsub4d(simd::permute4d<3, 3, 0, 0>(d), c, simd::mul4d(simd::permute4d<1, 1, 2, 2>(d), simd::permute4d<2, 3, 0, 1>(c)));
						const simd::reg4d xadj = simd::fmsub4d(ds3, a, simd::fmadd4d(b, simd::permute4d<0, 3, 0, 3>(dc), simd::mul4d(simd::permute4d<1, 0, 3, 2>(b), simd::permute4d<2, 1, 2, 1>(dc))));
						const simd::reg4d wadj = simd::fmsub4d(ds0, d, simd::fmadd4d(c, simd::permute4d<0, 3, 0, 3>(ab), simd::mul4d(simd::permute4d<1, 0, 3, 2>(c), simd::permute4d<2, 1, 2, 1>(ab))));
						const simd::reg4d yadj = simd::fmsub4d(ds1, c, simd::fmsub4d(d, simd::permute4d<3, 0, 3, 0>(ab), simd::mul4d(simd::permute4d<1, 0, 3, 2>(d), simd::permute4d<2, 1, 2, 1>(ab))));
						const simd::reg4d zadj = simd::fmsub4d(ds2, b, simd::fmsub4d(a, simd::permute4d<3, 0, 3, 0>(dc), simd::mul4d(simd::permute4d<1, 0, 3, 2>(a), simd::permute4d<2, 1, 2, 1>(dc))));
						const simd::reg4d trm = simd::mul4d(ab, simd::permute4d<0, 2, 1, 3>(dc));
						const simd::reg4d trs = simd::add4d(trm, simd::permute4d<1, 0, 3, 2>(trm));
						const simd::reg4d detm = simd::sub4d(simd::fmadd4d(ds0, ds3, simd::mul4d(ds1, ds2)), simd::add4d(trs, simd::permute4d<2, 3, 0, 1>(trs)));
						if (simd::extract4d_lo(detm) == static_cast<T>(0)) {
							return *this;
						}
						const simd::reg4d rdet = simd::div4d(simd::blend4d<0b0110>(simd::load1d4d(static_cast<T>(1)), simd::sub4d(simd::zero_reg4d(), simd::load1d4d(static_cast<T>(1)))), detm);
						const simd::reg4d xs = simd::mul4d(xadj, rdet);
						const simd::reg4d ys = simd::mul4d(yadj, rdet);
						const simd::reg4d zs = simd::mul4d(zadj, rdet);
						const simd::reg4d ws = simd::mul4d(wadj, rdet);
						simd::store4d(values[0], simd::blend4d<0b1100>(simd::permute4d<3, 1, 0, 0>(xs), simd::permute4d<0, 0, 3, 1>(ys)));
						simd::store4d(values[4], simd::blend4d<0b1100>(simd::permute4d<2, 0, 0, 0>(xs), simd::permute4d<0, 0, 2, 0>(ys)));
						simd::store4d(values[8], simd::blend4d<0b1100>(simd::permute4d<3, 1, 0, 0>(zs), simd::permute4d<0, 0, 3, 1>(ws)));
						simd::store4d(values[12], simd::blend4d<0b1100>(simd::permute4d<2, 0, 0, 0>(zs), simd::permute4d<0, 0, 2, 0>(ws)));
						return *this;
					#endif
					}
				}
			#endif
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

			for (auto i = 0u; i < 16u; i++) {
				values[i] = adjugate[i] / det;
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::transpose() noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					simd::store16f(values[0], simd::permutexvar16f(simd::load16f16f(values[0]), simd::create_mask_16i32<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>()));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg8d lo = simd::load8d8d(values[0]);
					const simd::reg8d hi = simd::load8d8d(values[8]);
					simd::store8d(values[0], simd::permutex2var8d(lo, hi, simd::create_mask_8i64<0, 4, 8, 12, 1, 5, 9, 13>()));
					simd::store8d(values[8], simd::permutex2var8d(lo, hi, simd::create_mask_8i64<2, 6, 10, 14, 3, 7, 11, 15>()));
					return *this;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					const simd::reg4f r0 = simd::load4f4f(values[0]);
					const simd::reg4f r1 = simd::load4f4f(values[4]);
					const simd::reg4f r2 = simd::load4f4f(values[8]);
					const simd::reg4f r3 = simd::load4f4f(values[12]);
					const simd::reg4f t0 = simd::unpacklo4f(r0, r1);
					const simd::reg4f t1 = simd::unpackhi4f(r0, r1);
					const simd::reg4f t2 = simd::unpacklo4f(r2, r3);
					const simd::reg4f t3 = simd::unpackhi4f(r2, r3);
					simd::store4f(values[0], simd::shuffle4f<0, 1, 0, 1>(t0, t2));
					simd::store4f(values[4], simd::shuffle4f<2, 3, 2, 3>(t0, t2));
					simd::store4f(values[8], simd::shuffle4f<0, 1, 0, 1>(t1, t3));
					simd::store4f(values[12], simd::shuffle4f<2, 3, 2, 3>(t1, t3));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg4d r0 = simd::load4d4d(values[0]);
					const simd::reg4d r1 = simd::load4d4d(values[4]);
					const simd::reg4d r2 = simd::load4d4d(values[8]);
					const simd::reg4d r3 = simd::load4d4d(values[12]);
					const simd::reg4d t0 = simd::unpacklo4d(r0, r1);
					const simd::reg4d t1 = simd::unpackhi4d(r0, r1);
					const simd::reg4d t2 = simd::unpacklo4d(r2, r3);
					const simd::reg4d t3 = simd::unpackhi4d(r2, r3);
					simd::store4d(values[0], simd::permute2f128_4d<0x20>(t0, t2));
					simd::store4d(values[4], simd::permute2f128_4d<0x20>(t1, t3));
					simd::store4d(values[8], simd::permute2f128_4d<0x31>(t0, t2));
					simd::store4d(values[12], simd::permute2f128_4d<0x31>(t1, t3));
					return *this;
				}
			}
		#endif
			const T transposed[16] = {
				values[0], values[4], values[8], values[12],
				values[1], values[5], values[9], values[13],
				values[2], values[6], values[10], values[14],
				values[3], values[7], values[11], values[15]
			};

			for (auto i = 0u; i < 16u; i++) {
				values[i] = transposed[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr bool mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator==(const mat4& matrix) const noexcept {
			for (auto i = 0u; i < 16u; i++) {
				if (values[i] != matrix.values[i]) {
					return false;
				}
			}
			return true;
		}

		template <typename T, std::size_t A>
		constexpr bool mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator!=(const mat4& matrix) const noexcept {
			for (auto i = 0u; i < 16u; i++) {
				if (values[i] != matrix.values[i]) {
					return true;
				}
			}
			return false;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator+=(const mat4& matrix) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					simd::store16f(values[0], simd::add16f(simd::load16f16f(values[0]), simd::load16f16f(matrix.values[0])));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store8d(values[0], simd::add8d(simd::load8d8d(values[0]), simd::load8d8d(matrix.values[0])));
					simd::store8d(values[8], simd::add8d(simd::load8d8d(values[8]), simd::load8d8d(matrix.values[8])));
					return *this;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					simd::store8f(values[0], simd::add8f(simd::load8f8f(values[0]), simd::load8f8f(matrix.values[0])));
					simd::store8f(values[8], simd::add8f(simd::load8f8f(values[8]), simd::load8f8f(matrix.values[8])));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store4d(values[0], simd::add4d(simd::load4d4d(values[0]), simd::load4d4d(matrix.values[0])));
					simd::store4d(values[4], simd::add4d(simd::load4d4d(values[4]), simd::load4d4d(matrix.values[4])));
					simd::store4d(values[8], simd::add4d(simd::load4d4d(values[8]), simd::load4d4d(matrix.values[8])));
					simd::store4d(values[12], simd::add4d(simd::load4d4d(values[12]), simd::load4d4d(matrix.values[12])));
					return *this;
				}
			}
		#endif
			for (auto i = 0u; i < 16u; i++) {
				values[i] += matrix.values[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator-=(const mat4& matrix) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					simd::store16f(values[0], simd::sub16f(simd::load16f16f(values[0]), simd::load16f16f(matrix.values[0])));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store8d(values[0], simd::sub8d(simd::load8d8d(values[0]), simd::load8d8d(matrix.values[0])));
					simd::store8d(values[8], simd::sub8d(simd::load8d8d(values[8]), simd::load8d8d(matrix.values[8])));
					return *this;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					simd::store8f(values[0], simd::sub8f(simd::load8f8f(values[0]), simd::load8f8f(matrix.values[0])));
					simd::store8f(values[8], simd::sub8f(simd::load8f8f(values[8]), simd::load8f8f(matrix.values[8])));
					return *this;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store4d(values[0], simd::sub4d(simd::load4d4d(values[0]), simd::load4d4d(matrix.values[0])));
					simd::store4d(values[4], simd::sub4d(simd::load4d4d(values[4]), simd::load4d4d(matrix.values[4])));
					simd::store4d(values[8], simd::sub4d(simd::load4d4d(values[8]), simd::load4d4d(matrix.values[8])));
					simd::store4d(values[12], simd::sub4d(simd::load4d4d(values[12]), simd::load4d4d(matrix.values[12])));
					return *this;
				}
			}
		#endif
			for (auto i = 0u; i < 16u; i++) {
				values[i] -= matrix.values[i];
			}
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>& mat4<T, A, std::enable_if_t<std::is_floating_point<T>::value>>::operator*=(const mat4& matrix) noexcept {
			*this = *this * matrix;
			return *this;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator+(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				mat4<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					simd::store16f(simd_result.values[0], simd::add16f(simd::load16f16f(left.values[0]), simd::load16f16f(right.values[0])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store8d(simd_result.values[0], simd::add8d(simd::load8d8d(left.values[0]), simd::load8d8d(right.values[0])));
					simd::store8d(simd_result.values[8], simd::add8d(simd::load8d8d(left.values[8]), simd::load8d8d(right.values[8])));
					return simd_result;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				mat4<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					simd::store8f(simd_result.values[0], simd::add8f(simd::load8f8f(left.values[0]), simd::load8f8f(right.values[0])));
					simd::store8f(simd_result.values[8], simd::add8f(simd::load8f8f(left.values[8]), simd::load8f8f(right.values[8])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store4d(simd_result.values[0], simd::add4d(simd::load4d4d(left.values[0]), simd::load4d4d(right.values[0])));
					simd::store4d(simd_result.values[4], simd::add4d(simd::load4d4d(left.values[4]), simd::load4d4d(right.values[4])));
					simd::store4d(simd_result.values[8], simd::add4d(simd::load4d4d(left.values[8]), simd::load4d4d(right.values[8])));
					simd::store4d(simd_result.values[12], simd::add4d(simd::load4d4d(left.values[12]), simd::load4d4d(right.values[12])));
					return simd_result;
				}
			}
		#endif
			mat4<T, A> result;
			for (auto i = 0u; i < 16u; i++) {
				result.values[i] = left.values[i] + right.values[i];
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator-(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				mat4<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					simd::store16f(simd_result.values[0], simd::sub16f(simd::load16f16f(left.values[0]), simd::load16f16f(right.values[0])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store8d(simd_result.values[0], simd::sub8d(simd::load8d8d(left.values[0]), simd::load8d8d(right.values[0])));
					simd::store8d(simd_result.values[8], simd::sub8d(simd::load8d8d(left.values[8]), simd::load8d8d(right.values[8])));
					return simd_result;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				mat4<T, A> simd_result;
				if constexpr (std::is_same_v<T, float>) {
					simd::store8f(simd_result.values[0], simd::sub8f(simd::load8f8f(left.values[0]), simd::load8f8f(right.values[0])));
					simd::store8f(simd_result.values[8], simd::sub8f(simd::load8f8f(left.values[8]), simd::load8f8f(right.values[8])));
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					simd::store4d(simd_result.values[0], simd::sub4d(simd::load4d4d(left.values[0]), simd::load4d4d(right.values[0])));
					simd::store4d(simd_result.values[4], simd::sub4d(simd::load4d4d(left.values[4]), simd::load4d4d(right.values[4])));
					simd::store4d(simd_result.values[8], simd::sub4d(simd::load4d4d(left.values[8]), simd::load4d4d(right.values[8])));
					simd::store4d(simd_result.values[12], simd::sub4d(simd::load4d4d(left.values[12]), simd::load4d4d(right.values[12])));
					return simd_result;
				}
			}
		#endif
			mat4<T, A> result;
			for (auto i = 0u; i < 16u; i++) {
				result.values[i] = left.values[i] - right.values[i];
			}
			return result;
		}

		template <typename T, std::size_t A>
		constexpr mat4<T, A> operator*(const mat4<T, A>& left, const mat4<T, A>& right) noexcept {
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					const simd::reg16f l = simd::load16f16f(left.values[0]);
					simd::reg16f acc = simd::mul16f(simd::permutexvar16f(l, simd::create_mask_16i32<0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12>()), simd::broadcast4f16f(right.values[0]));
					acc = simd::fmadd16f(simd::permutexvar16f(l, simd::create_mask_16i32<1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13>()), simd::broadcast4f16f(right.values[4]), acc);
					acc = simd::fmadd16f(simd::permutexvar16f(l, simd::create_mask_16i32<2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14>()), simd::broadcast4f16f(right.values[8]), acc);
					acc = simd::fmadd16f(simd::permutexvar16f(l, simd::create_mask_16i32<3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15>()), simd::broadcast4f16f(right.values[12]), acc);
					mat4<T, A> simd_result;
					simd::store16f(simd_result.values[0], acc);
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg8d l0 = simd::load8d8d(left.values[0]);
					const simd::reg8d l1 = simd::load8d8d(left.values[8]);
					// Right rows broadcast directly from memory.
					const simd::reg8d b0 = simd::broadcast4d8d(right.values[0]);
					const simd::reg8d b1 = simd::broadcast4d8d(right.values[4]);
					const simd::reg8d b2 = simd::broadcast4d8d(right.values[8]);
					const simd::reg8d b3 = simd::broadcast4d8d(right.values[12]);
					// Each column's left multiplier broadcast for the row pair held in each half.
					const simd::reg8d l00 = simd::permutexvar8d(l0, simd::create_mask_8i64<0, 0, 0, 0, 4, 4, 4, 4>());
					const simd::reg8d l01 = simd::permutexvar8d(l0, simd::create_mask_8i64<1, 1, 1, 1, 5, 5, 5, 5>());
					const simd::reg8d l02 = simd::permutexvar8d(l0, simd::create_mask_8i64<2, 2, 2, 2, 6, 6, 6, 6>());
					const simd::reg8d l03 = simd::permutexvar8d(l0, simd::create_mask_8i64<3, 3, 3, 3, 7, 7, 7, 7>());
					const simd::reg8d l10 = simd::permutexvar8d(l1, simd::create_mask_8i64<0, 0, 0, 0, 4, 4, 4, 4>());
					const simd::reg8d l11 = simd::permutexvar8d(l1, simd::create_mask_8i64<1, 1, 1, 1, 5, 5, 5, 5>());
					const simd::reg8d l12 = simd::permutexvar8d(l1, simd::create_mask_8i64<2, 2, 2, 2, 6, 6, 6, 6>());
					const simd::reg8d l13 = simd::permutexvar8d(l1, simd::create_mask_8i64<3, 3, 3, 3, 7, 7, 7, 7>());
					const simd::reg8d acc0 = simd::fmadd8d(l03, b3, simd::fmadd8d(l02, b2, simd::fmadd8d(l01, b1, simd::mul8d(l00, b0))));
					const simd::reg8d acc1 = simd::fmadd8d(l13, b3, simd::fmadd8d(l12, b2, simd::fmadd8d(l11, b1, simd::mul8d(l10, b0))));
					mat4<T, A> simd_result;
					simd::store8d(simd_result.values[0], acc0);
					simd::store8d(simd_result.values[8], acc1);
					return simd_result;
				}
			}
		#elif defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					// Two rows per accumulator: lanes 0-3 hold one row, lanes 4-7 the next.
					const simd::reg8f lb0 = simd::load8f8f(left.values[0]);
					const simd::reg8f lb1 = simd::load8f8f(left.values[8]);
					const simd::reg8f rb0 = simd::load8f8f(right.values[0]);
					const simd::reg8f rb1 = simd::load8f8f(right.values[8]);
					// Each right row duplicated across both halves, reused by both accumulators.
					const simd::reg8f rr0 = simd::permutevar8f(rb0, simd::create_mask_8i32<0, 1, 2, 3, 0, 1, 2, 3>());
					const simd::reg8f rr1 = simd::permutevar8f(rb0, simd::create_mask_8i32<4, 5, 6, 7, 4, 5, 6, 7>());
					const simd::reg8f rr2 = simd::permutevar8f(rb1, simd::create_mask_8i32<0, 1, 2, 3, 0, 1, 2, 3>());
					const simd::reg8f rr3 = simd::permutevar8f(rb1, simd::create_mask_8i32<4, 5, 6, 7, 4, 5, 6, 7>());
					// Each left column broadcast within its row's half.
					const simd::reg8f la0 = simd::permutevar8f(lb0, simd::create_mask_8i32<0, 0, 0, 0, 4, 4, 4, 4>());
					const simd::reg8f la1 = simd::permutevar8f(lb0, simd::create_mask_8i32<1, 1, 1, 1, 5, 5, 5, 5>());
					const simd::reg8f la2 = simd::permutevar8f(lb0, simd::create_mask_8i32<2, 2, 2, 2, 6, 6, 6, 6>());
					const simd::reg8f la3 = simd::permutevar8f(lb0, simd::create_mask_8i32<3, 3, 3, 3, 7, 7, 7, 7>());
					const simd::reg8f lb_0 = simd::permutevar8f(lb1, simd::create_mask_8i32<0, 0, 0, 0, 4, 4, 4, 4>());
					const simd::reg8f lb_1 = simd::permutevar8f(lb1, simd::create_mask_8i32<1, 1, 1, 1, 5, 5, 5, 5>());
					const simd::reg8f lb_2 = simd::permutevar8f(lb1, simd::create_mask_8i32<2, 2, 2, 2, 6, 6, 6, 6>());
					const simd::reg8f lb_3 = simd::permutevar8f(lb1, simd::create_mask_8i32<3, 3, 3, 3, 7, 7, 7, 7>());
					const simd::reg8f acc0 = simd::fmadd8f(la3, rr3, simd::fmadd8f(la2, rr2, simd::fmadd8f(la1, rr1, simd::mul8f(la0, rr0))));
					const simd::reg8f acc1 = simd::fmadd8f(lb_3, rr3, simd::fmadd8f(lb_2, rr2, simd::fmadd8f(lb_1, rr1, simd::mul8f(lb_0, rr0))));
					mat4<T, A> simd_result;
					simd::store8f(simd_result.values[0], acc0);
					simd::store8f(simd_result.values[8], acc1);
					return simd_result;
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg4d l0 = simd::load4d4d(left.values[0]);
					const simd::reg4d l1 = simd::load4d4d(left.values[4]);
					const simd::reg4d l2 = simd::load4d4d(left.values[8]);
					const simd::reg4d l3 = simd::load4d4d(left.values[12]);
					const simd::reg4d r0 = simd::load4d4d(right.values[0]);
					const simd::reg4d r1 = simd::load4d4d(right.values[4]);
					const simd::reg4d r2 = simd::load4d4d(right.values[8]);
					const simd::reg4d r3 = simd::load4d4d(right.values[12]);
					mat4<T, A> simd_result;
					simd::store4d(simd_result.values[0], simd::fmadd4d(simd::permute4d<3, 3, 3, 3>(l0), r3, simd::fmadd4d(simd::permute4d<2, 2, 2, 2>(l0), r2, simd::fmadd4d(simd::permute4d<1, 1, 1, 1>(l0), r1, simd::mul4d(simd::permute4d<0, 0, 0, 0>(l0), r0)))));
					simd::store4d(simd_result.values[4], simd::fmadd4d(simd::permute4d<3, 3, 3, 3>(l1), r3, simd::fmadd4d(simd::permute4d<2, 2, 2, 2>(l1), r2, simd::fmadd4d(simd::permute4d<1, 1, 1, 1>(l1), r1, simd::mul4d(simd::permute4d<0, 0, 0, 0>(l1), r0)))));
					simd::store4d(simd_result.values[8], simd::fmadd4d(simd::permute4d<3, 3, 3, 3>(l2), r3, simd::fmadd4d(simd::permute4d<2, 2, 2, 2>(l2), r2, simd::fmadd4d(simd::permute4d<1, 1, 1, 1>(l2), r1, simd::mul4d(simd::permute4d<0, 0, 0, 0>(l2), r0)))));
					simd::store4d(simd_result.values[12], simd::fmadd4d(simd::permute4d<3, 3, 3, 3>(l3), r3, simd::fmadd4d(simd::permute4d<2, 2, 2, 2>(l3), r2, simd::fmadd4d(simd::permute4d<1, 1, 1, 1>(l3), r1, simd::mul4d(simd::permute4d<0, 0, 0, 0>(l3), r0)))));
					return simd_result;
				}
			}
		#endif
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
		#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4f), "vec3 must be aligned for SIMD loads");
					const simd::reg16f v = simd::blend16fmask(
						simd::broadcast4f16f(simd::load4f4f(vector.x)),
						simd::load1f16f(static_cast<T>(1)),
						static_cast<simd::mask16>(0x8888)
					);
					simd::reg16f p = simd::mul16f(simd::load16f16f(matrix.values[0]), v);
					p = simd::add16f(p, simd::shuffle16f<2, 3, 0, 1>(p, p));
					p = simd::add16f(p, simd::shuffle16f<1, 0, 3, 2>(p, p));
					alignas(alignof(simd::reg16f)) T sums[16];
					simd::store16f(sums[0], p);
					return {sums[0], sums[4], sums[8]};
				}
			}
		#endif
			#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					const T* m = matrix.values;
					const simd::reg4f vx = simd::load1f4f_lo(vector.x);
					const simd::reg4f vy = simd::load1f4f_lo(vector.y);
					const simd::reg4f vz = simd::load1f4f_lo(vector.z);
					const simd::reg4f r0 = simd::fmadd4f_lo(
						simd::load1f4f_lo(m[2]),
						vz,
						simd::fmadd4f_lo(
							simd::load1f4f_lo(m[1]),
							vy,
							simd::fmadd4f_lo(simd::load1f4f_lo(m[0]), vx, simd::load1f4f_lo(m[3]))
						)
					);
					const simd::reg4f r1 = simd::fmadd4f_lo(
						simd::load1f4f_lo(m[6]),
						vz,
						simd::fmadd4f_lo(
							simd::load1f4f_lo(m[5]),
							vy,
							simd::fmadd4f_lo(simd::load1f4f_lo(m[4]), vx, simd::load1f4f_lo(m[7]))
						)
					);
					const simd::reg4f r2 = simd::fmadd4f_lo(
						simd::load1f4f_lo(m[10]),
						vz,
						simd::fmadd4f_lo(
							simd::load1f4f_lo(m[9]),
							vy,
							simd::fmadd4f_lo(simd::load1f4f_lo(m[8]), vx, simd::load1f4f_lo(m[11]))
						)
					);
					return {simd::extract4f_lo(r0), simd::extract4f_lo(r1), simd::extract4f_lo(r2)};
				} else if constexpr (std::is_same_v<T, double>) {
				#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
					static_assert(alignof(vec3<T, VA>) >= alignof(simd::reg4d), "vec3 must be aligned for SIMD loads");
					const simd::reg4d v = simd::blend4d<0b1000>(
						simd::load4d4d(vector.x),
						simd::load1d4d(static_cast<T>(1)));
					const simd::reg4d p0 = simd::mul4d(simd::load4d4d(matrix.values[0]), v);
					const simd::reg4d p1 = simd::mul4d(simd::load4d4d(matrix.values[4]), v);
					const simd::reg4d p2 = simd::mul4d(simd::load4d4d(matrix.values[8]), v);
					const simd::reg4d s0 = simd::add4d(p0, simd::permute4d<2, 3, 0, 1>(p0));
					const simd::reg4d s1 = simd::add4d(p1, simd::permute4d<2, 3, 0, 1>(p1));
					const simd::reg4d s2 = simd::add4d(p2, simd::permute4d<2, 3, 0, 1>(p2));
					const simd::reg4d t0 = simd::add4d(s0, simd::permute4d<1, 0, 3, 2>(s0));
					const simd::reg4d t1 = simd::add4d(s1, simd::permute4d<1, 0, 3, 2>(s1));
					const simd::reg4d t2 = simd::add4d(s2, simd::permute4d<1, 0, 3, 2>(s2));
					return {simd::extract4d_lo(t0), simd::extract4d_lo(t1), simd::extract4d_lo(t2)};
				#else
					const T* m = matrix.values;
					const simd::reg2d vx = simd::load1d2d_lo(vector.x);
					const simd::reg2d vy = simd::load1d2d_lo(vector.y);
					const simd::reg2d vz = simd::load1d2d_lo(vector.z);
					const simd::reg2d r0 = simd::fmadd2d_lo(
						simd::load1d2d_lo(m[2]),
						vz,
						simd::fmadd2d_lo(
							simd::load1d2d_lo(m[1]),
							vy,
							simd::fmadd2d_lo(simd::load1d2d_lo(m[0]), vx, simd::load1d2d_lo(m[3]))
						)
					);
					const simd::reg2d r1 = simd::fmadd2d_lo(
						simd::load1d2d_lo(m[6]),
						vz,
						simd::fmadd2d_lo(
							simd::load1d2d_lo(m[5]),
							vy,
							simd::fmadd2d_lo(simd::load1d2d_lo(m[4]), vx, simd::load1d2d_lo(m[7]))
						)
					);
					const simd::reg2d r2 = simd::fmadd2d_lo(
						simd::load1d2d_lo(m[10]),
						vz,
						simd::fmadd2d_lo(
							simd::load1d2d_lo(m[9]),
							vy,
							simd::fmadd2d_lo(simd::load1d2d_lo(m[8]), vx, simd::load1d2d_lo(m[11]))
						)
					);
					return {simd::extract2d_lo(r0), simd::extract2d_lo(r1), simd::extract2d_lo(r2)};
				#endif
				}
			}
		#endif
			return {
				matrix.values[0] * vector.x + matrix.values[1] * vector.y + matrix.values[2] * vector.z + matrix.values[3],
				matrix.values[4] * vector.x + matrix.values[5] * vector.y + matrix.values[6] * vector.z + matrix.values[7],
				matrix.values[8] * vector.x + matrix.values[9] * vector.y + matrix.values[10] * vector.z + matrix.values[11]
			};
		}

		template <typename T, std::size_t A, std::size_t VA>
		constexpr vec3<T, VA> operator*(const vec3<T, VA>& vector, const mat4<T, A>& matrix) noexcept {
			// Row-vector multiply: result_j = [x, y, z, 1] * column_j.
		#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
			if (!std::is_constant_evaluated()) {
				if constexpr (std::is_same_v<T, float>) {
					const simd::reg4f res = simd::fmadd4f(
						simd::load4f4f(matrix.values[8]), simd::load1f4f(vector.z),
						simd::fmadd4f(
							simd::load4f4f(matrix.values[4]), simd::load1f4f(vector.y),
							simd::fmadd4f(
								simd::load4f4f(matrix.values[0]),
								simd::load1f4f(vector.x),
								simd::load4f4f(matrix.values[12])
							)
						)
					);
					vec3<T, VA> out;
					simd::store4f(out.x, res);
					return out;
				} else if constexpr (std::is_same_v<T, double>) {
					const simd::reg4d res = simd::fmadd4d(
						simd::load4d4d(matrix.values[8]), simd::load1d4d(vector.z),
						simd::fmadd4d(
							simd::load4d4d(matrix.values[4]),
							simd::load1d4d(vector.y),
							simd::fmadd4d(
								simd::load4d4d(matrix.values[0]),
								simd::load1d4d(vector.x),
								simd::load4d4d(matrix.values[12])
							)
						)
					);
					vec3<T, VA> out;
					simd::store4d(out.x, res);
					return out;
				}
			}
		#endif
			return {
				matrix.values[0] * vector.x + matrix.values[4] * vector.y + matrix.values[8] * vector.z + matrix.values[12],
				matrix.values[1] * vector.x + matrix.values[5] * vector.y + matrix.values[9] * vector.z + matrix.values[13],
				matrix.values[2] * vector.x + matrix.values[6] * vector.y + matrix.values[10] * vector.z + matrix.values[14]
			};
		}
	} // namespace math
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
