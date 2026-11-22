#pragma once
#if defined(__AVX__)

#define BENCH_BOX_SIMD_REG2D
#define BENCH_BOX_SIMD_REG4F
#define BENCH_BOX_SIMD_REG8F
#define BENCH_BOX_SIMD_REG4I32
#define BENCH_BOX_SIMD_REG2I64

#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		using reg2d = __m128d;

		using reg4f = __m128;
		using reg8f = __m256;

		using reg4i32 = __m128i;
		using reg2i64 = __m128i;
	} // namespace simd
} // namespace bench_box

#include "avx_f64.h"
#include "avx_f32.h"
#include "avx_i32.h"
#include "avx_i64.h"

#endif
