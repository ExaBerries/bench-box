#pragma once
#if defined(__AVX512VL__)

#define BENCH_BOX_SIMD_REG4D_VL

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d permutex2var4d(const reg4d& a, const reg4d& b, const reg4i64& idx) noexcept {
			return _mm256_permutex2var_pd(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d permutex2var2d(const reg2d& a, const reg2d& b, const reg2i64& idx) noexcept {
			return _mm_permutex2var_pd(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f permutex2var8f(const reg8f& a, const reg8f& b, const reg8i32& idx) noexcept {
			return _mm256_permutex2var_ps(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f permutex2var4f(const reg4f& a, const reg4f& b, const reg4i32& idx) noexcept {
			return _mm_permutex2var_ps(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 permutex2var8i32(const reg8i32& a, const reg8i32& b, const reg8i32& idx) noexcept {
			return _mm256_permutex2var_epi32(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 permutex2var4i32(const reg4i32& a, const reg4i32& b, const reg4i32& idx) noexcept {
			return _mm_permutex2var_epi32(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 permutexvar4i64(const reg4i64& a, const reg4i64& idx) noexcept {
			return _mm256_permutexvar_epi64(idx, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 permutex2var4i64(const reg4i64& a, const reg4i64& b, const reg4i64& idx) noexcept {
			return _mm256_permutex2var_epi64(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 permutex2var2i64(const reg2i64& a, const reg2i64& b, const reg2i64& idx) noexcept {
			return _mm_permutex2var_epi64(a, idx, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 abs4i64(const reg4i64& a) noexcept {
			return _mm256_abs_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 min4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_min_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 max4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_max_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 minu4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_min_epu64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 maxu4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_max_epu64(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 sra4i64(const reg4i64& a) noexcept {
			return _mm256_srai_epi64(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 srafv4i64(const reg4i64& a, const reg4i64& count) noexcept {
			return _mm256_srav_epi64(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cvt4i64_to_4i32(const reg4i64& a) noexcept {
			return _mm256_cvtepi64_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cvtu4i64_to_4i32(const reg4i64& a) noexcept {
			return _mm256_cvtusepi64_epi32(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
