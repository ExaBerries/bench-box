#pragma once

#include <bench_box/isa/isa_macro.h>

#if defined(BENCH_BOX_SIMD_USE_INTRIN_FMA)
	#if !defined(__AVX2__)
		#error "BENCH_BOX_SIMD_USE_INTRIN_FMA requires AVX2"
	#endif
	#if !defined(__FMA__) && !defined(_MSC_VER)
		#error "BENCH_BOX_SIMD_USE_INTRIN_FMA requires FMA"
	#endif
#endif

#if defined(BENCH_BOX_SIMD_USE_AVX512VL)
	#if !defined(__AVX512F__) || !defined(__AVX512VL__)
		#error "BENCH_BOX_SIMD_USE_AVX512VL requires AVX512F and AVX512VL"
	#endif
#endif

#if defined(BENCH_BOX_ISA_X86) || defined(BENCH_BOX_ISA_X86_64)
	#include <immintrin.h>

	#if defined(__AVX__)
		#include <bench_box/isa/x86/avx.h>

		#if defined(__AVX2__)
			#include <bench_box/isa/x86/avx2.h>
		#endif

		#if defined(__AVX512F__)
			#include <bench_box/isa/x86/avx512.h>
		#endif

		#include <bench_box/isa/x86/avx512_vl.h>
		#include <bench_box/isa/x86/avx512_vl_dq.h>
	#endif

	#include <bench_box/isa/x86/misc.h>
#endif
