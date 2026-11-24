#pragma once
#if defined(__AVX512DQ__)
#define BENCH_BOX_SIMD_AVX512DQ

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d and8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_and_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d andnot8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_andnot_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d or8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_or_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d xor8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_xor_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f and16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_and_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f andnot16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_andnot_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f or16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_or_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f xor16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_xor_ps(a, b);
		}

		// low 64 bits of the lane-wise 64-bit product
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 mul8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_mullo_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 mulu8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return mul8i64(a, b); // the low 64 bits are identical for signed and unsigned products
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 mullo8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return mul8i64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvt8f_to_8i64(const reg8f& a) noexcept {
			return _mm512_cvtps_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvtt8f_to_8i64(const reg8f& a) noexcept {
			return _mm512_cvttps_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvt8d_to_8i64(const reg8d& a) noexcept {
			return _mm512_cvtpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvtt8d_to_8i64(const reg8d& a) noexcept {
			return _mm512_cvttpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvt8f_to_u8i64(const reg8f& a) noexcept {
			return _mm512_cvtps_epu64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 cvt8d_to_u8i64(const reg8d& a) noexcept {
			return _mm512_cvtpd_epu64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cvt8i64_to_8f(const reg8i64& a) noexcept {
			return _mm512_cvtepi64_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cvtu8i64_to_8f(const reg8i64& a) noexcept {
			return _mm512_cvtepu64_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d cvt8i64_to_8d(const reg8i64& a) noexcept {
			return _mm512_cvtepi64_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d cvtu8i64_to_8d(const reg8i64& a) noexcept {
			return _mm512_cvtepu64_pd(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
