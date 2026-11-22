#pragma once
#if defined(__AVX512VPOPCNTDQ__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg16i32 popcnt16i32(const reg16i32& a) noexcept {
			return _mm512_popcnt_epi32(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 popcnt8i64(const reg8i64& a) noexcept {
			return _mm512_popcnt_epi64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
