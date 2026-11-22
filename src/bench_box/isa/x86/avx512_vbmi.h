#pragma once
#if defined(__AVX512VBMI__)

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		[[gnu::always_inline, nodiscard]] inline reg64i8 permutexvar64i8(const reg64i8& a, const reg64i8& idx) noexcept {
			return _mm512_permutexvar_epi8(idx, a);
		}

		[[gnu::always_inline, nodiscard]] inline reg64i8 permutex2var64i8(const reg64i8& a, const reg64i8& b, const reg64i8& idx) noexcept {
			return _mm512_permutex2var_epi8(a, idx, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg64i8 multishift64i8(const reg64i8& ctrl, const reg64i8& tables) noexcept {
			return _mm512_multishift_epi64_epi8(ctrl, tables);
		}
	} // namespace simd
} // namespace bench_box

#endif
