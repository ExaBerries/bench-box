#pragma once
#if defined(__AVX2__)

#define BENCH_BOX_SIMD_REG4I64
#define BENCH_BOX_SIMD_REG8I32

// MSVC does not define __FMA__, but /arch:AVX2 implies FMA
#if defined(__FMA__) || defined(_MSC_VER)
	#define BENCH_BOX_SIMD_FMA
#endif

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		using reg4i64 = __m256i;

		using reg8i32 = __m256i;
	} // namespace simd
} // namespace bench_box

#include "avx2_f64.h"
#include "avx2_f32.h"
#include "avx2_i32.h"
#include "avx2_i64.h"

#endif
