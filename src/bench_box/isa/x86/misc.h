#pragma once
#include <immintrin.h>

namespace bench_box {
	template <typename T>
	BENCH_BOX_ALWAYS_INLINE void prefetch_read(const T* ptr) noexcept {
		_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
	}

	template <typename T>
	BENCH_BOX_ALWAYS_INLINE void prefetch_write(const T* ptr) noexcept {
		_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_ET0);
	}

	template <typename T>
	BENCH_BOX_ALWAYS_INLINE void prefetch_nta(const T* ptr) noexcept {
		_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
	}

	// automatically round denormals to zero
	inline void round_denormals() noexcept {
		_mm_setcsr(_mm_getcsr() | 0x8000 | 0x0040);
	}
} // namespace bench_box
