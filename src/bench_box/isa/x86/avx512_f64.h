#pragma once

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg8d zero_reg8d() noexcept {
			return _mm512_setzero_pd();
		}

		[[gnu::always_inline, nodiscard]] inline double extract8d_lo(const reg8d& a) noexcept {
			return _mm512_cvtsd_f64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d load8d8d(const double& data) noexcept {
			return _mm512_load_pd(&data);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d load4d8d_both(const double& data) noexcept {
			return _mm512_broadcast_f64x4(_mm256_load_pd(&data));
		}

		[[gnu::always_inline, nodiscard]] inline reg8d load4d8d_lower(const double& data) noexcept {
			return _mm512_castpd256_pd512(_mm256_load_pd(&data));
		}

		[[gnu::always_inline, nodiscard]] inline reg8d load1d8d(const double& value) noexcept {
			return _mm512_set1_pd(value);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d mask_load8d8d(const reg8d& src, const double& data, const mask8& k) noexcept {
			return _mm512_mask_load_pd(src, k, &data);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d maskz_load8d8d(const double& data, const mask8& k) noexcept {
			return _mm512_maskz_load_pd(k, &data);
		}

		[[gnu::always_inline]] inline void store8d(double& out, const reg8d& reg) noexcept {
			_mm512_store_pd(&out, reg);
		}

		[[gnu::always_inline]] inline void mask_store8d(double& out, const mask8& k, const reg8d& reg) noexcept {
			_mm512_mask_store_pd(&out, k, reg);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d reg4d8d(const reg4d& a) noexcept {
			return _mm512_castpd256_pd512(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d reg4d8d(const reg4d& a, const reg4d& b) noexcept {
			return _mm512_insertf64x4(_mm512_castpd256_pd512(a), b, 1);
		}

		[[gnu::always_inline, nodiscard]] inline reg4d reg8d4d(const reg8d& a) noexcept {
			return _mm512_castpd512_pd256(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d add8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_add_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d sub8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_sub_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d mul8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_mul_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d div8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_div_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d sqrt8d(const reg8d& a) noexcept {
			return _mm512_sqrt_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d min8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_min_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d max8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_max_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_lt8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_le8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_eq8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_neq8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_gt8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(b, a, _CMP_LT_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_ge8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_cmp_pd_mask(b, a, _CMP_LE_OS);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d blend8dmask(const reg8d& a, const reg8d& b, const mask8& mask) noexcept {
			return _mm512_mask_blend_pd(mask, a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmadd8d(const reg8d& a, const reg8d& b, const reg8d& c) noexcept {
			return _mm512_fmadd_pd(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmsub8d(const reg8d& a, const reg8d& b, const reg8d& c) noexcept {
			return _mm512_fmsub_pd(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmnadd8d(const reg8d& a, const reg8d& b, const reg8d& c) noexcept {
			return _mm512_fnmadd_pd(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmnsub8d(const reg8d& a, const reg8d& b, const reg8d& c) noexcept {
			return _mm512_fnmsub_pd(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmadd8dmask3(const reg8d& a, const reg8d& b, const reg8d& c, const mask8& m) noexcept {
			return _mm512_mask3_fmadd_pd(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmsub8dmask3(const reg8d& a, const reg8d& b, const reg8d& c, const mask8& m) noexcept {
			return _mm512_mask3_fmsub_pd(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmnadd8dmask3(const reg8d& a, const reg8d& b, const reg8d& c, const mask8& m) noexcept {
			return _mm512_mask3_fnmadd_pd(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d fmnsub8dmask3(const reg8d& a, const reg8d& b, const reg8d& c, const mask8& m) noexcept {
			return _mm512_mask3_fnmsub_pd(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d permutexvar8d(const reg8d& a, const reg8i64& mask) noexcept {
			return _mm512_permutexvar_pd(mask, a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d permutex2var8d(const reg8d& a, const reg8d& b, const reg8i64& idx) noexcept {
			return _mm512_permutex2var_pd(a, idx, b);
		}

		[[gnu::always_inline, nodiscard]] inline double reduce_add8d(const reg8d& a) noexcept {
			return _mm512_reduce_add_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline double reduce_min8d(const reg8d& a) noexcept {
			return _mm512_reduce_min_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline double reduce_max8d(const reg8d& a) noexcept {
			return _mm512_reduce_max_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d cvt8i32_to_8d(const reg8i32& a) noexcept;

		[[gnu::always_inline, nodiscard]] inline reg8d cvt8f_to_8d(const reg8f& a) noexcept {
			return _mm512_cvtps_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d abs8d(const reg8d& a) noexcept {
			return _mm512_abs_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d neg8d(const reg8d& a) noexcept {
			return _mm512_sub_pd(zero_reg8d(), a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d cvt8i32_to_8d(const reg8i32& a) noexcept {
			return _mm512_cvtepi32_pd(a);
		}
	} // namespace simd
} // namespace bench_box
