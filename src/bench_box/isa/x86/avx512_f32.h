#pragma once

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg16f zero_reg16f() noexcept {
			return _mm512_setzero_ps();
		}

		[[gnu::always_inline, nodiscard]] inline float extract16f_lo(const reg16f& a) noexcept {
			return _mm512_cvtss_f32(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f load16f16f(const float& data) noexcept {
			return _mm512_load_ps(&data);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f load8f16f_both(const float& data) noexcept {
			const __m256i lower = _mm256_castps_si256(_mm256_load_ps(&data));
			return _mm512_castsi512_ps(_mm512_inserti64x4(_mm512_castsi256_si512(lower), lower, 1));
		}

		[[gnu::always_inline, nodiscard]] inline reg16f load8f16f_lower(const float& data) noexcept {
			return _mm512_castps256_ps512(_mm256_load_ps(&data));
		}

		[[gnu::always_inline, nodiscard]] inline reg16f load1f16f(const float& value) noexcept {
			return _mm512_set1_ps(value);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f mask_load16f16f(const reg16f& src, const float& data, const mask16& k) noexcept {
			return _mm512_mask_load_ps(src, k, &data);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f maskz_load16f16f(const float& data, const mask16& k) noexcept {
			return _mm512_maskz_load_ps(k, &data);
		}

		[[gnu::always_inline]] inline void store16f(float& out, const reg16f& reg) noexcept {
			_mm512_store_ps(&out, reg);
		}

		[[gnu::always_inline]] inline void mask_store16f(float& out, const mask16& k, const reg16f& reg) noexcept {
			_mm512_mask_store_ps(&out, k, reg);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f reg8f16f(const reg8f& a) noexcept {
			return _mm512_castps256_ps512(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f reg8f16f(const reg8f& a, const reg8f& b) noexcept {
			return _mm512_castsi512_ps(_mm512_inserti64x4(_mm512_castps_si512(_mm512_castps256_ps512(a)), _mm256_castps_si256(b), 1));
		}

		[[gnu::always_inline, nodiscard]] inline reg8f reg16f8f(const reg16f& a) noexcept {
			return _mm512_castps512_ps256(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f add16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_add_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f sub16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_sub_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f mul16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_mul_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f div16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_div_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f sqrt16f(const reg16f& a) noexcept {
			return _mm512_sqrt_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f abs16f(const reg16f& a) noexcept {
			return _mm512_abs_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f neg16f(const reg16f& a) noexcept {
			return _mm512_sub_ps(zero_reg16f(), a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f min16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_min_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f max16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_max_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_lt16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_le16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_eq16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_neq16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_gt16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(b, a, _CMP_LT_OS);
		}

		[[gnu::always_inline, nodiscard]] inline mask16 cmp_ge16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_cmp_ps_mask(b, a, _CMP_LE_OS);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f blend16fmask(const reg16f& a, const reg16f& b, const mask16& mask) noexcept {
			return _mm512_mask_blend_ps(mask, a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmadd16f(const reg16f& a, const reg16f& b, const reg16f& c) noexcept {
			return _mm512_fmadd_ps(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmsub16f(const reg16f& a, const reg16f& b, const reg16f& c) noexcept {
			return _mm512_fmsub_ps(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmnadd16f(const reg16f& a, const reg16f& b, const reg16f& c) noexcept {
			return _mm512_fnmadd_ps(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmnsub16f(const reg16f& a, const reg16f& b, const reg16f& c) noexcept {
			return _mm512_fnmsub_ps(a, b, c);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmadd16fmask3(const reg16f& a, const reg16f& b, const reg16f& c, const mask16& m) noexcept {
			return _mm512_mask3_fmadd_ps(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmsub16fmask3(const reg16f& a, const reg16f& b, const reg16f& c, const mask16& m) noexcept {
			return _mm512_mask3_fmsub_ps(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmnadd16fmask3(const reg16f& a, const reg16f& b, const reg16f& c, const mask16& m) noexcept {
			return _mm512_mask3_fnmadd_ps(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f fmnsub16fmask3(const reg16f& a, const reg16f& b, const reg16f& c, const mask16& m) noexcept {
			return _mm512_mask3_fnmsub_ps(a, b, c, m);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f permutexvar16f(const reg16f& a, const reg16i32& mask) noexcept {
			return _mm512_permutexvar_ps(mask, a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f permutex2var16f(const reg16f& a, const reg16f& b, const reg16i32& idx) noexcept {
			return _mm512_permutex2var_ps(a, idx, b);
		}

		[[gnu::always_inline, nodiscard]] inline float reduce_add16f(const reg16f& a) noexcept {
			return _mm512_reduce_add_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline float reduce_min16f(const reg16f& a) noexcept {
			return _mm512_reduce_min_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline float reduce_max16f(const reg16f& a) noexcept {
			return _mm512_reduce_max_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8f cvt8d_to_8f(const reg8d& a) noexcept {
			return _mm512_cvtpd_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f cvt16i32_to_16f(const reg16i32& a) noexcept {
			return _mm512_cvtepi32_ps(a);
		}
	} // namespace simd
} // namespace bench_box
