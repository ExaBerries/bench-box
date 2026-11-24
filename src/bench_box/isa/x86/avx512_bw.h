#pragma once
#if defined(__AVX512BW__)
#define BENCH_BOX_SIMD_AVX512BW

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 load64i8(const int8_t& data) noexcept {
			return _mm512_load_si512(reinterpret_cast<const reg64i8*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 broadcast64i8(const int8_t& value) noexcept {
			return _mm512_set1_epi8(static_cast<signed char>(value));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 mask_load64i8(const reg64i8& src, const int8_t& data, const mask64& k) noexcept {
			return _mm512_mask_loadu_epi8(src, k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 maskz_load64i8(const int8_t& data, const mask64& k) noexcept {
			return _mm512_maskz_loadu_epi8(k, &data);
		}

		BENCH_BOX_ALWAYS_INLINE void store64i8(int8_t& out, const reg64i8& reg) noexcept {
			_mm512_store_si512(reinterpret_cast<reg64i8*>(&out), reg);
		}

		BENCH_BOX_ALWAYS_INLINE void mask_store64i8(int8_t& out, const mask64& k, const reg64i8& reg) noexcept {
			_mm512_mask_storeu_epi8(&out, k, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 zero_reg64i8() noexcept {
			return _mm512_setzero_si512();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 add64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_add_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 sub64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_sub_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 add_sat64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_adds_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 sub_sat64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_subs_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 add_sat_u64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_adds_epu8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 sub_sat_u64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_subs_epu8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 avg_u64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_avg_epu8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 min64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_min_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 max64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_max_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 minu64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_min_epu8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 maxu64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_max_epu8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 abs64i8(const reg64i8& a) noexcept {
			return _mm512_abs_epi8(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maddubs64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_maddubs_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_lt64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_le64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_eq64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_neq64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_NE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_gt64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmp_ge64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmpu_lt64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmpu_le64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmpu_gt64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 cmpu_ge64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 test64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_test_epi8_mask(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 testn64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_testn_epi8_mask(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask64 movepi64i8(const reg64i8& a) noexcept {
			return _mm512_movepi8_mask(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 blend64i8mask(const reg64i8& a, const reg64i8& b, const mask64& mask) noexcept {
			return _mm512_mask_blend_epi8(mask, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 unpacklo64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_unpacklo_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 unpackhi64i8(const reg64i8& a, const reg64i8& b) noexcept {
			return _mm512_unpackhi_epi8(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 load32i16(const int16_t& data) noexcept {
			return _mm512_load_si512(reinterpret_cast<const reg32i16*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 broadcast32i16(const int16_t& value) noexcept {
			return _mm512_set1_epi16(static_cast<short>(value));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mask_load32i16(const reg32i16& src, const int16_t& data, const mask32& k) noexcept {
			return _mm512_mask_loadu_epi16(src, k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maskz_load32i16(const int16_t& data, const mask32& k) noexcept {
			return _mm512_maskz_loadu_epi16(k, &data);
		}

		BENCH_BOX_ALWAYS_INLINE void store32i16(int16_t& out, const reg32i16& reg) noexcept {
			_mm512_store_si512(reinterpret_cast<reg32i16*>(&out), reg);
		}

		BENCH_BOX_ALWAYS_INLINE void mask_store32i16(int16_t& out, const mask32& k, const reg32i16& reg) noexcept {
			_mm512_mask_storeu_epi16(&out, k, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 zero_reg32i16() noexcept {
			return _mm512_setzero_si512();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 add32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_add_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sub32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_sub_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 add_sat32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_adds_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sub_sat32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_subs_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 add_sat_u32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_adds_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sub_sat_u32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_subs_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 avg_u32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_avg_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 min32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_min_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 max32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_max_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 minu32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_min_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maxu32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_max_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 abs32i16(const reg32i16& a) noexcept {
			return _mm512_abs_epi16(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mullo32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_mullo_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mulhi32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_mulhi_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mulhi_u32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_mulhi_epu16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 madd32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_madd_epi16(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sl32i16(const reg32i16& a) noexcept {
			return _mm512_slli_epi16(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sr32i16(const reg32i16& a) noexcept {
			return _mm512_srli_epi16(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 sra32i16(const reg32i16& a) noexcept {
			return _mm512_srai_epi16(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 slv32i16(const reg32i16& a, const reg32i16& count) noexcept {
			return _mm512_sllv_epi16(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 srfv32i16(const reg32i16& a, const reg32i16& count) noexcept {
			return _mm512_srlv_epi16(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 srafv32i16(const reg32i16& a, const reg32i16& count) noexcept {
			return _mm512_srav_epi16(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_lt32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_le32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_eq32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_EQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_neq32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_NE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_gt32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmp_ge32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmpu_lt32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_LT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmpu_le32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_LE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmpu_gt32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_NLE);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 cmpu_ge32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_NLT);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 test32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_test_epi16_mask(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 testn32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_testn_epi16_mask(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE mask32 movepi32i16(const reg32i16& a) noexcept {
			return _mm512_movepi16_mask(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 blend32i16mask(const reg32i16& a, const reg32i16& b, const mask32& mask) noexcept {
			return _mm512_mask_blend_epi16(mask, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 unpacklo32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_unpacklo_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 unpackhi32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_unpackhi_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 permutexvar32i16(const reg32i16& a, const reg32i16& idx) noexcept {
			return _mm512_permutexvar_epi16(idx, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 permutex2var32i16(const reg32i16& a, const reg32i16& b, const reg32i16& idx) noexcept {
			return _mm512_permutex2var_epi16(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 packs16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_packs_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 packus16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_packus_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 packs32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_packs_epi16(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 packus32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_packus_epi16(a, b);
		}
	} // namespace simd
} // namespace bench_box

#endif
