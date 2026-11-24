#pragma once
#include "avx_f64.h"

namespace bench_box {
	namespace simd {
		template <uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d permute4d(const reg4d& a) noexcept {
			return _mm256_permute4x64_pd(a, _MM_SHUFFLE(m3, m2, m1, m0));
		}

		#if defined(BENCH_BOX_SIMD_FMA)
			// scalar fma into lane 0 only
			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d fmadd2d_lo(const reg2d& a, const reg2d& b, const reg2d& c) noexcept {
				return _mm_fmadd_sd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d fmadd2d(const reg2d& a, const reg2d& b, const reg2d& c) noexcept {
				return _mm_fmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmadd4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmsub4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fmsub_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmnadd4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fnmadd_pd(a, b, c);
			}

			[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d fmnsub4d(const reg4d& a, const reg4d& b, const reg4d& c) noexcept {
				return _mm256_fnmsub_pd(a, b, c);
			}
		#endif
	} // namespace simd
} // namespace bench_box
