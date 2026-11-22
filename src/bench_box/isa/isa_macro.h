#pragma once

#if defined(__x86_64__) || defined(_M_X64)
	#define BENCH_BOX_ISA_X86_64
#elif defined(__i386__) || defined(_M_IX86)
	#define BENCH_BOX_ISA_X86
#elif defined(__aarch64__) || defined(_M_ARM64)
	#define BENCH_BOX_ISA_ARM64
#endif

// MSVC ignores [[gnu::always_inline]]; use __forceinline to make sure the
// intrinsic wrappers are inlined there as well.
#if defined(_MSC_VER)
	#define BENCH_BOX_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
	#define BENCH_BOX_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
	#define BENCH_BOX_ALWAYS_INLINE inline
#endif
