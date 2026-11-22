#pragma once

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d load4d4d(const double& data) noexcept {
			return _mm256_load_pd(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d load1d4d(const double& value) noexcept {
			return _mm256_set1_pd(value);
		}

		BENCH_BOX_ALWAYS_INLINE void store4d(double& out, const reg4d& reg) noexcept {
			_mm256_store_pd(&out, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d zero_reg4d() noexcept {
			return _mm256_setzero_pd();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d reg4d2d_lo(const reg4d& a) noexcept {
			return _mm256_castpd256_pd128(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d reg4d2d_hi(const reg4d& a) noexcept {
			return _mm256_extractf128_pd(a, 1);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE double extract4d_lo(const reg4d& a) noexcept {
			return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d add4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_add_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d sub4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_sub_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d mul4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_mul_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d div4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_div_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d sqrt4d(const reg4d& a) noexcept {
			return _mm256_sqrt_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d min4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_min_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d max4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_max_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d abs4d(const reg4d& a) noexcept {
			return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d neg4d(const reg4d& a) noexcept {
			return _mm256_xor_pd(a, _mm256_set1_pd(-0.0));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d and4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_and_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d andnot4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_andnot_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d or4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_or_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d xor4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_xor_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_lt4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(a, b, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_le4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(a, b, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_eq4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_gt4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(b, a, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_ge4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(b, a, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cmp_neq4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ);
		}

		template <uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d permute4d(const reg4d& a) noexcept {
			return _mm256_permute4x64_pd(a, _MM_SHUFFLE(m3, m2, m1, m0));
		}

		template <uint32_t mask>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d blend4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_blend_pd(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d unpacklo4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_unpacklo_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d unpackhi4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_unpackhi_pd(a, b);
		}

		template <int32_t immediate>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d permute2f128_4d(const reg4d& a, const reg4d& b) noexcept {
			return _mm256_permute2f128_pd(a, b, immediate);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d blendv4d(const reg4d& a, const reg4d& b, const reg4d& mask) noexcept {
			return _mm256_castps_pd(_mm256_blendv_ps(_mm256_castpd_ps(a), _mm256_castpd_ps(b), _mm256_castpd_ps(mask)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cvt4f_to_4d(const reg4f& a) noexcept {
			return _mm256_cvtps_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d reg2d4d(const reg2d& a) noexcept {
			return _mm256_insertf128_pd(_mm256_castpd128_pd256(a), a, 1);
		}

		#if defined(BENCH_BOX_SIMD_FMA)
			// scalar fma into lane 0 only
			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d fmadd2d_lo(const reg2d& a, const reg2d& b, const reg2d& c) noexcept {
				return _mm_fmadd_sd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d fmadd2d(const reg2d& a, const reg2d& b, const reg2d& c) noexcept {
				return _mm_fmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmadd4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmsub4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fmsub_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmnadd4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fnmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmnsub4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fnmsub_pd(a, b, c);
			}
		#endif
	} // namespace simd
} // namespace bench_box
