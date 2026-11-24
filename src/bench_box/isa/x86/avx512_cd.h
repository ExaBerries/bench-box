#pragma once
#if defined(__AVX512CD__)
	#define BENCH_BOX_SIMD_AVX512CD

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 conflict16i32(const reg16i32& a) noexcept {
			return _mm512_conflict_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 conflict8i64(const reg8i64& a) noexcept {
			return _mm512_conflict_epi64(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
