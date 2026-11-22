#pragma once
#if defined(__AVX512VL__) && defined(__AVX512DQ__)

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		// Native 64-bit lane multiply. The AVX/AVX2 forms in avx_i64.h and
		// avx2_i64.h are compound fallbacks because those ISAs lack it.
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 mullo2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_mullo_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mullo4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_mullo_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cvt2i64_to_2d(const reg2i64& a) noexcept {
			return _mm_cvtepi64_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cvtu2i64_to_2d(const reg2i64& a) noexcept {
			return _mm_cvtepu64_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cvt2d_to_2i64(const reg2d& a) noexcept {
			return _mm_cvtpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cvtt2d_to_2i64(const reg2d& a) noexcept {
			return _mm_cvttpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cvt2d_to_u2i64(const reg2d& a) noexcept {
			return _mm_cvtpd_epu64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cvt4i64_to_4d(const reg4i64& a) noexcept {
			return _mm256_cvtepi64_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d cvtu4i64_to_4d(const reg4i64& a) noexcept {
			return _mm256_cvtepu64_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cvt4d_to_4i64(const reg4d& a) noexcept {
			return _mm256_cvtpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cvtt4d_to_4i64(const reg4d& a) noexcept {
			return _mm256_cvttpd_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cvt4d_to_u4i64(const reg4d& a) noexcept {
			return _mm256_cvtpd_epu64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
