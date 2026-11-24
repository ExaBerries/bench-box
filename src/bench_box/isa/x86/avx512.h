#pragma once
#if defined(__AVX512F__)

#define BENCH_BOX_SIMD_AVX512F
#define BENCH_BOX_SIMD_REG8D
#define BENCH_BOX_SIMD_REG8I64
#define BENCH_BOX_SIMD_REG16F
#define BENCH_BOX_SIMD_REG16I32
#define BENCH_BOX_SIMD_REG64I8
#define BENCH_BOX_SIMD_REG32I16

#include <bench_box/isa/isa_macro.h>
#include <cstdint>
#include <immintrin.h>

namespace bench_box {
	namespace simd {
		using reg8d = __m512d;
		using reg8i64 = __m512i;
		using mask8 = __mmask8;

		using reg16f = __m512;
		using reg16i32 = __m512i;
		using mask16 = __mmask16;

		using reg64i8 = __m512i;
		using reg32i16 = __m512i;
		using mask64 = __mmask64;
		using mask32 = __mmask32;
	} // namespace simd
} // namespace bench_box

#include "avx512_f64.h"
#include "avx512_f32.h"
#include "avx512_i64.h"
#include "avx512_i32.h"
#include "avx512_gather.h"
#include "avx512_scatter.h"
#include "avx512_cd.h"
#include "avx512_dq.h"
#include "avx512_vnni.h"
#include "avx512_bw.h"
#include "avx512_vbmi.h"
#include "avx512_vbmi2.h"
#include "avx512_vpopcntdq.h"
#include "avx512_bitalg.h"

#endif
