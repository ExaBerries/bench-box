#pragma once
#if defined(__SSE2__)

#define BENCH_BOX_SIMD_REG4F
#define BENCH_BOX_SIMD_REG4I32

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		using reg4f = __m128;
		using reg4i32 = __m128i;
	} // namespace simd
} // namespace bench_box

#include "sse2_f32.h"
#include "sse2_i32.h"

#endif
