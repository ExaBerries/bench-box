#pragma once

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg8f permutevar8f(const reg8f& a, const reg8i32& idx) noexcept {
			return _mm256_permutevar8x32_ps(a, idx);
		}

		[[gnu::always_inline, nodiscard]] inline reg4f cvt4d_to_4f(const reg4d& a) noexcept {
			return _mm256_cvtpd_ps(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8f cvt8i32_to_8f(const reg8i32& a) noexcept {
			return _mm256_cvtepi32_ps(a);
		}

		#if defined(BENCH_BOX_SIMD_FMA)
			[[gnu::always_inline, nodiscard]] inline reg4f fmadd4f(const reg4f& a, const reg4f& b, const reg4f& c) noexcept {
				return _mm_fmadd_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg4f fmsub4f(const reg4f& a, const reg4f& b, const reg4f& c) noexcept {
				return _mm_fmsub_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg4f fmnadd4f(const reg4f& a, const reg4f& b, const reg4f& c) noexcept {
				return _mm_fnmadd_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg4f fmnsub4f(const reg4f& a, const reg4f& b, const reg4f& c) noexcept {
				return _mm_fnmsub_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg8f fmadd8f(const reg8f& a, const reg8f& b, const reg8f& c) noexcept {
				return _mm256_fmadd_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg8f fmsub8f(const reg8f& a, const reg8f& b, const reg8f& c) noexcept {
				return _mm256_fmsub_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg8f fmnadd8f(const reg8f& a, const reg8f& b, const reg8f& c) noexcept {
				return _mm256_fnmadd_ps(a, b, c);
			}

			[[gnu::always_inline, nodiscard]] inline reg8f fmnsub8f(const reg8f& a, const reg8f& b, const reg8f& c) noexcept {
				return _mm256_fnmsub_ps(a, b, c);
			}
		#endif
	} // namespace simd
} // namespace bench_box
