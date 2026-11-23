#pragma once

namespace bench_box {
	namespace simd {
		template <int32_t m0, int32_t m1, int32_t m2, int32_t m3, int32_t m4, int32_t m5, int32_t m6, int32_t m7, int32_t m8, int32_t m9, int32_t m10, int32_t m11, int32_t m12, int32_t m13, int32_t m14, int32_t m15>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 create_mask_16i32() noexcept {
			alignas(alignof(reg16i32)) static constexpr int32_t mask_data[16] = {
				m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
			};
			return _mm512_load_si512(reinterpret_cast<const reg16i32*>(mask_data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 load16i32(const int32_t& data) noexcept {
			return _mm512_load_si512(reinterpret_cast<const reg16i32*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 loadu16i32(const int32_t& data) noexcept {
			return _mm512_loadu_si512(reinterpret_cast<const reg16i32*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 broadcast16i32(const int32_t& value) noexcept {
			return _mm512_set1_epi32(static_cast<int>(value));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 mask_load16i32(const reg16i32& src, const int32_t& data, const mask16& k) noexcept {
			return _mm512_mask_load_epi32(src, k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 maskz_load16i32(const int32_t& data, const mask16& k) noexcept {
			return _mm512_maskz_load_epi32(k, &data);
		}

		BENCH_BOX_ALWAYS_INLINE void store16i32(int32_t& out, const reg16i32& reg) noexcept {
			_mm512_store_si512(reinterpret_cast<reg16i32*>(&out), reg);
		}

		BENCH_BOX_ALWAYS_INLINE void storeu16i32(int32_t& out, const reg16i32& reg) noexcept {
			_mm512_storeu_si512(reinterpret_cast<reg16i32*>(&out), reg);
		}

		BENCH_BOX_ALWAYS_INLINE void mask_store16i32(int32_t& out, const mask16& k, const reg16i32& reg) noexcept {
			_mm512_mask_store_epi32(&out, k, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 zero_reg16i32() noexcept {
			return _mm512_setzero_si512();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t extract16i32_lo(const reg16i32& a) noexcept {
			return _mm512_cvtsi512_si32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 add16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_add_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 sub16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_sub_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 mul16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_mullo_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 abs16i32(const reg16i32& a) noexcept {
			return _mm512_abs_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t reduce_and16i32(const reg16i32& a) noexcept {
			return _mm512_reduce_and_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t reduce_or16i32(const reg16i32& a) noexcept {
			return _mm512_reduce_or_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 min16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_min_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 max16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_max_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 minu16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_min_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 maxu16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_max_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 and16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_and_si512(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 andnot16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_andnot_si512(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 or16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_or_si512(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 xor16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_xor_si512(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 sl16i32(const reg16i32& a) noexcept {
			return _mm512_slli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 sr16i32(const reg16i32& a) noexcept {
			return _mm512_srli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 sra16i32(const reg16i32& a) noexcept {
			return _mm512_srai_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 slv16i32(const reg16i32& a, const reg16i32& count) noexcept {
			return _mm512_sllv_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 srfv16i32(const reg16i32& a, const reg16i32& count) noexcept {
			return _mm512_srlv_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 srafv16i32(const reg16i32& a, const reg16i32& count) noexcept {
			return _mm512_srav_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_lt16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_le16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_eq16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_neq16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_gt16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmp_ge16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmpu_lt16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmpu_le16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmpu_gt16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 cmpu_ge16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 blend16i32mask(const reg16i32& a, const reg16i32& b, const mask16& mask) noexcept {
			return _mm512_mask_blend_epi32(mask, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 permutexvar16i32(const reg16i32& a, const reg16i32& idx) noexcept {
			return _mm512_permutexvar_epi32(idx, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 permutex2var16i32(const reg16i32& a, const reg16i32& b, const reg16i32& idx) noexcept {
			return _mm512_permutex2var_epi32(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 cvt16f_to_16i32(const reg16f& a) noexcept {
			return _mm512_cvtps_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 cvtt16f_to_16i32(const reg16f& a) noexcept {
			return _mm512_cvttps_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cvt8i64_to_8i32(const reg8i64& a) noexcept {
			return _mm512_cvtepi64_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cvtu8i64_to_8i32(const reg8i64& a) noexcept {
			return _mm512_cvtusepi64_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t reduce_add16i32(const reg16i32& a) noexcept {
			return _mm512_reduce_add_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t reduce_min16i32(const reg16i32& a) noexcept {
			return _mm512_reduce_min_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t reduce_max16i32(const reg16i32& a) noexcept {
			return _mm512_reduce_max_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 and16mask(const mask16& a, const mask16& b) noexcept {
			return static_cast<mask16>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 or16mask(const mask16& a, const mask16& b) noexcept {
			return static_cast<mask16>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask16 not16mask(const mask16& a) noexcept {
			return static_cast<mask16>(~static_cast<uint32_t>(a));
		}
	} // namespace simd
} // namespace bench_box
