#pragma once
#if defined(__AVX512VBMI2__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 shldi32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_shldi_epi16(a, b, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 shldi16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_shldi_epi32(a, b, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 shldi8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_shldi_epi64(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 shldv32i16(const reg32i16& a, const reg32i16& b, const reg32i16& count) noexcept {
			return _mm512_shldv_epi16(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 shldv16i32(const reg16i32& a, const reg16i32& b, const reg16i32& count) noexcept {
			return _mm512_shldv_epi32(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 shldv8i64(const reg8i64& a, const reg8i64& b, const reg8i64& count) noexcept {
			return _mm512_shldv_epi64(a, b, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 shrdi32i16(const reg32i16& a, const reg32i16& b) noexcept {
			return _mm512_shrdi_epi16(a, b, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 shrdi16i32(const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_shrdi_epi32(a, b, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 shrdi8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_shrdi_epi64(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 shrdv32i16(const reg32i16& a, const reg32i16& b, const reg32i16& count) noexcept {
			return _mm512_shrdv_epi16(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 shrdv16i32(const reg16i32& a, const reg16i32& b, const reg16i32& count) noexcept {
			return _mm512_shrdv_epi32(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 shrdv8i64(const reg8i64& a, const reg8i64& b, const reg8i64& count) noexcept {
			return _mm512_shrdv_epi64(a, b, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 maskz_compress64i8(const reg64i8& a, const mask64& k) noexcept {
			return _mm512_maskz_compress_epi8(k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 mask_compress64i8(const reg64i8& src, const reg64i8& a, const mask64& k) noexcept {
			return _mm512_mask_compress_epi8(src, k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maskz_compress32i16(const reg32i16& a, const mask32& k) noexcept {
			return _mm512_maskz_compress_epi16(k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mask_compress32i16(const reg32i16& src, const reg32i16& a, const mask32& k) noexcept {
			return _mm512_mask_compress_epi16(src, k, a);
		}

		BENCH_BOX_ALWAYS_INLINE void mask_compressstoreu64i8(int8_t& out, const mask64& k, const reg64i8& reg) noexcept {
			_mm512_mask_compressstoreu_epi8(&out, k, reg);
		}

		BENCH_BOX_ALWAYS_INLINE void mask_compressstoreu32i16(int16_t& out, const mask32& k, const reg32i16& reg) noexcept {
			_mm512_mask_compressstoreu_epi16(&out, k, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 maskz_expand64i8(const reg64i8& a, const mask64& k) noexcept {
			return _mm512_maskz_expand_epi8(k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 mask_expand64i8(const reg64i8& src, const reg64i8& a, const mask64& k) noexcept {
			return _mm512_mask_expand_epi8(src, k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maskz_expand32i16(const reg32i16& a, const mask32& k) noexcept {
			return _mm512_maskz_expand_epi16(k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mask_expand32i16(const reg32i16& src, const reg32i16& a, const mask32& k) noexcept {
			return _mm512_mask_expand_epi16(src, k, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 maskz_expandloadu64i8(const int8_t& data, const mask64& k) noexcept {
			return _mm512_maskz_expandloadu_epi8(k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg64i8 mask_expandloadu64i8(const reg64i8& src, const int8_t& data, const mask64& k) noexcept {
			return _mm512_mask_expandloadu_epi8(src, k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 maskz_expandloadu32i16(const int16_t& data, const mask32& k) noexcept {
			return _mm512_maskz_expandloadu_epi16(k, &data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg32i16 mask_expandloadu32i16(const reg32i16& src, const int16_t& data, const mask32& k) noexcept {
			return _mm512_mask_expandloadu_epi16(src, k, &data);
		}
	} // namespace simd
} // namespace bench_box

#endif
