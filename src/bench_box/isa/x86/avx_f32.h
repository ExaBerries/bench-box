#pragma once

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f load4f4f(const float& data) noexcept {
			return _mm_load_ps(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f load1f4f(const float& value) noexcept {
			return _mm_set1_ps(value);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f load1f4f_lo(const float& value) noexcept {
			return _mm_load_ss(&value);
		}

		BENCH_BOX_ALWAYS_INLINE void store4f(float& out, const reg4f& reg) noexcept {
			_mm_store_ps(&out, reg);
		}

		BENCH_BOX_ALWAYS_INLINE void store4f_lo(float& out, const reg4f& reg) noexcept {
			_mm_store_ss(&out, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f zero_reg4f() noexcept {
			return _mm_setzero_ps();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE float extract4f_lo(const reg4f& a) noexcept {
			return _mm_cvtss_f32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f add4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_add_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f sub4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_sub_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f mul4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_mul_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f div4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_div_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f sqrt4f(const reg4f& a) noexcept {
			return _mm_sqrt_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f min4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_min_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f max4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_max_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f abs4f(const reg4f& a) noexcept {
			return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f neg4f(const reg4f& a) noexcept {
			return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f and4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_and_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f andnot4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_andnot_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f or4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_or_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f xor4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_xor_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_lt4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmplt_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_le4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmple_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_eq4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmpeq_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_gt4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmplt_ps(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_ge4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmple_ps(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cmp_neq4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_cmpneq_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f blendv4f(const reg4f& a, const reg4f& b, const reg4f& mask) noexcept {
			return _mm_blendv_ps(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f blendv8f(const reg8f& a, const reg8f& b, const reg8f& mask) noexcept {
			return _mm256_blendv_ps(a, b, mask);
		}

		template <uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f permute4f(const reg4f& a) noexcept {
			return _mm_permute_ps(a, _MM_SHUFFLE(m3, m2, m1, m0));
		}

		template <uint32_t w, uint32_t x, uint32_t y, uint32_t z>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f shuffle4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(z, y, x, w));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f unpacklo4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_unpacklo_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f unpackhi4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_unpackhi_ps(a, b);
		}

		template <uint32_t mask>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f blend4f(const reg4f& a, const reg4f& b) noexcept {
			return _mm_blend_ps(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f load8f8f(const float& data) noexcept {
			return _mm256_load_ps(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f load1f8f(const float& value) noexcept {
			return _mm256_set1_ps(value);
		}

		BENCH_BOX_ALWAYS_INLINE void store8f(float& out, const reg8f& reg) noexcept {
			_mm256_store_ps(&out, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f zero_reg8f() noexcept {
			return _mm256_setzero_ps();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f add8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_add_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f sub8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_sub_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f mul8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_mul_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f div8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_div_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f sqrt8f(const reg8f& a) noexcept {
			return _mm256_sqrt_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f min8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_min_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f max8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_max_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f abs8f(const reg8f& a) noexcept {
			return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f neg8f(const reg8f& a) noexcept {
			return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f and8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_and_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f andnot8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_andnot_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f or8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_or_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f xor8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_xor_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_lt8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_le8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_eq8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_gt8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(b, a, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_ge8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(b, a, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_neq8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
		}

		template <uint32_t mask>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f blend8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_blend_ps(a, b, mask);
		}
	} // namespace simd
} // namespace bench_box
