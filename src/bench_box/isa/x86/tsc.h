#pragma once
#include <bench_box/isa/isa_macro.h>
#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
#include <cstdint>
#include <immintrin.h>
#include <bench_box/isa/x86/fences.h>

namespace bench_box {
	[[nodiscard]] inline uint64_t read_tsc() noexcept {
		return __rdtsc();
	}

	[[nodiscard]] inline uint64_t read_tscp() noexcept {
		return __rdtscp(nullptr);
	}

	[[nodiscard]] inline uint64_t fence_rdtsc() noexcept {
		lfence();
		auto tsc = read_tsc();
		lfence();
		return tsc;
	}

	[[nodiscard]] inline uint64_t fence_rdtscp() noexcept {
		auto tsc = read_tscp();
		lfence();
		return tsc;
	}
} // namespace bench_box
#endif
