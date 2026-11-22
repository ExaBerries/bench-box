#pragma once
#if defined(__AVX512VL__) && defined(__AVX512DQ__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg2d cvt2i64_to_2d(const reg2i64& a) noexcept {
			return _mm_cvtepi64_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg2d cvtu2i64_to_2d(const reg2i64& a) noexcept {
			return _mm_cvtepu64_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg2i64 cvt2d_to_2i64(const reg2d& a) noexcept {
			return _mm_cvtpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg2i64 cvtt2d_to_2i64(const reg2d& a) noexcept {
			return _mm_cvttpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg2i64 cvt2d_to_u2i64(const reg2d& a) noexcept {
			return _mm_cvtpd_epu64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg4d cvt4i64_to_4d(const reg4i64& a) noexcept {
			return _mm256_cvtepi64_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg4d cvtu4i64_to_4d(const reg4i64& a) noexcept {
			return _mm256_cvtepu64_pd(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg4i64 cvt4d_to_4i64(const reg4d& a) noexcept {
			return _mm256_cvtpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg4i64 cvtt4d_to_4i64(const reg4d& a) noexcept {
			return _mm256_cvttpd_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg4i64 cvt4d_to_u4i64(const reg4d& a) noexcept {
			return _mm256_cvtpd_epu64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
