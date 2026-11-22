#pragma once
#if defined(__AVX512DQ__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg8d and8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_and_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d andnot8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_andnot_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d or8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_or_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d xor8d(const reg8d& a, const reg8d& b) noexcept {
			return _mm512_xor_pd(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f and16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_and_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f andnot16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_andnot_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f or16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_or_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg16f xor16f(const reg16f& a, const reg16f& b) noexcept {
			return _mm512_xor_ps(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 mullo8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_mullo_epi64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvt8f_to_8i64(const reg8f& a) noexcept {
			return _mm512_cvtps_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvtt8f_to_8i64(const reg8f& a) noexcept {
			return _mm512_cvttps_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvt8d_to_8i64(const reg8d& a) noexcept {
			return _mm512_cvtpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvtt8d_to_8i64(const reg8d& a) noexcept {
			return _mm512_cvttpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvt8f_to_u8i64(const reg8f& a) noexcept {
			return _mm512_cvtps_epu64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvt8d_to_u8i64(const reg8d& a) noexcept {
			return _mm512_cvtpd_epu64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8f cvt8i64_to_8f(const reg8i64& a) noexcept {
			return _mm512_cvtepi64_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8f cvtu8i64_to_8f(const reg8i64& a) noexcept {
			return _mm512_cvtepu64_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d cvt8i64_to_8d(const reg8i64& a) noexcept {
			return _mm512_cvtepi64_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8d cvtu8i64_to_8d(const reg8i64& a) noexcept {
			return _mm512_cvtepu64_pd(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
