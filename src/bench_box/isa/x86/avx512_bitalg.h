#pragma once
#if defined(__AVX512BITALG__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg64i8 popcnt64i8(const reg64i8& a) noexcept {
			return _mm512_popcnt_epi8(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg32i16 popcnt32i16(const reg32i16& a) noexcept {
			return _mm512_popcnt_epi16(a);
		}
	} // namespace simd
} // namespace bench_box

#endif
