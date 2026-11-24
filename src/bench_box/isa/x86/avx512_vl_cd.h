#pragma once
#if defined(BENCH_BOX_SIMD_AVX512CD)

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 conflict8i32(const reg8i32& a) noexcept {
			return _mm256_conflict_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 conflict4i32(const reg4i32& a) noexcept {
			return _mm_conflict_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 conflict4i64(const reg4i64& a) noexcept {
			return _mm256_conflict_epi64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 conflict2i64(const reg2i64& a) noexcept {
			return _mm_conflict_epi64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
