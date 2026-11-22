#pragma once
#if defined(__AVX512VNNI__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusd16i32(const reg16i32& acc, const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_dpbusd_epi32(acc, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusds16i32(const reg16i32& acc, const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_dpbusds_epi32(acc, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssd16i32(const reg16i32& acc, const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_dpwssd_epi32(acc, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssds16i32(const reg16i32& acc, const reg16i32& a, const reg16i32& b) noexcept {
			return _mm512_dpwssds_epi32(acc, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusd16i32mask(const reg16i32& acc, const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_mask_dpbusd_epi32(acc, k, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusds16i32mask(const reg16i32& acc, const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_mask_dpbusds_epi32(acc, k, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssd16i32mask(const reg16i32& acc, const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_mask_dpwssd_epi32(acc, k, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssds16i32mask(const reg16i32& acc, const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_mask_dpwssds_epi32(acc, k, a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusd16i32maskz(const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_maskz_dpbusd_epi32(k, zero_reg16i32(), a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpbusds16i32maskz(const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_maskz_dpbusds_epi32(k, zero_reg16i32(), a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssd16i32maskz(const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_maskz_dpwssd_epi32(k, zero_reg16i32(), a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 dpwssds16i32maskz(const reg16i32& a, const reg16i32& b, const mask16& k) noexcept {
			return _mm512_maskz_dpwssds_epi32(k, zero_reg16i32(), a, b);
		}
	} // namespace simd
} // namespace bench_box

#endif
