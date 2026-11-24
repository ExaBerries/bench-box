#pragma once

#include <bench_box/isa/isa_macro.h>

#if defined(BENCH_BOX_ISA_X86) || defined(BENCH_BOX_ISA_X86_64)
	#include <immintrin.h>

	#include <bench_box/isa/x86/sse2.h>
	#include <bench_box/isa/x86/avx.h>
	#include <bench_box/isa/x86/avx2.h>
	#include <bench_box/isa/x86/avx512.h>
	#include <bench_box/isa/x86/avx512_vl.h>
	#include <bench_box/isa/x86/avx512_vl_dq.h>
	#include <bench_box/isa/x86/misc.h>
#endif
