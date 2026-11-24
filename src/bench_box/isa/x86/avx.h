#pragma once
#if defined(__AVX__)

#define BENCH_BOX_SIMD_AVX
#define BENCH_BOX_SIMD_REG4D
#define BENCH_BOX_SIMD_REG8F
#define BENCH_BOX_SIMD_REG2I64

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		using reg4d = __m256d;
		using reg8f = __m256;
		using reg2i64 = __m128i;
	} // namespace simd
} // namespace bench_box

#include "sse2.h"

#include "avx_f64.h"
#include "avx_f32.h"
#include "avx_i64.h"

#endif
