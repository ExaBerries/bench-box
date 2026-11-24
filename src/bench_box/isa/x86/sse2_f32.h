#pragma once

namespace bench_box {
	namespace simd {

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f create4f(float m0, float m1, float m2, float m3) noexcept {
			return _mm_set_ps(m3, m2, m1, m0);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f load4f4f(const float& data) noexcept {
			return _mm_load_ps(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f loadu4f4f(const float& data) noexcept {
			return _mm_loadu_ps(&data);
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

		BENCH_BOX_ALWAYS_INLINE void storeu4f(float& out, const reg4f& reg) noexcept {
			_mm_storeu_ps(&out, reg);
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
			#if defined(BENCH_BOX_SIMD_SSE4_1)
				return _mm_blendv_ps(a, b, mask);
			#else
				return or4f(and4f(mask, b), andnot4f(mask, a));
			#endif
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f floor4f(const reg4f& a) noexcept {
			#if defined(BENCH_BOX_SIMD_SSE4_1)
				return _mm_floor_ps(a);
			#else
				const reg4f truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
				const reg4f adjust = and4f(cmp_lt4f(a, truncated), _mm_set1_ps(1.0f));
				return sub4f(truncated, adjust);
			#endif
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cast4i32_to_4f(const reg4i32& a) noexcept {
			return _mm_castsi128_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cast4f_to_4i32(const reg4f& a) noexcept {
			return _mm_castps_si128(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f cvt4i32_to_4f(const reg4i32& a) noexcept {
			return _mm_cvtepi32_ps(a);
		}

		template <uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f permute4f(const reg4f& a) noexcept {
			return _mm_shuffle_ps(a, a, _MM_SHUFFLE(m3, m2, m1, m0));
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
			#if defined(BENCH_BOX_SIMD_SSE4_1)
				return _mm_blend_ps(a, b, static_cast<int>(mask));
			#else
				const reg4i32 mask_reg = _mm_set_epi32(
					((mask & 0x8u) != 0u) ? -1 : 0,
					((mask & 0x4u) != 0u) ? -1 : 0,
					((mask & 0x2u) != 0u) ? -1 : 0,
					((mask & 0x1u) != 0u) ? -1 : 0
				);
				const reg4f select = cast4i32_to_4f(mask_reg);
				return or4f(and4f(select, b), andnot4f(select, a));
			#endif
		}
	} // namespace simd
} // namespace bench_box
