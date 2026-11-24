#pragma once
#if defined(__AVX512VPOPCNTDQ__)
#define BENCH_BOX_SIMD_AVX512VPOPCNTDQ

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 popcnt16i32(const reg16i32& a) noexcept {
			return _mm512_popcnt_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 popcnt8i64(const reg8i64& a) noexcept {
			return _mm512_popcnt_epi64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
