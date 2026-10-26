#pragma once
#include <bench_box/isa/isa_macro.h>
#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
#include <string>
#include <cstdint>
#include <array>

namespace bench_box {
	struct cpuid_result {
		int32_t eax;
		int32_t ebx;
		int32_t ecx;
		int32_t edx;
	};

	enum struct avx10_vector_length_t {
		AVX_10_LENGTH_UNKNOWN,
		AVX_10_128,
		AVX_10_256,
		AVX_10_512
	};

	struct x86_extensions {
		bool mmx = false;
		bool sse = false;
		bool sse2 = false;
		bool sse3 = false;
		bool ssse3 = false;
		bool sse4_1 = false;
		bool sse4_2 = false;
		bool sse4a = false;
		bool popcnt = false;
		bool aes = false;
		bool rdrand = false;
		bool bmi1 = false;
		bool bmi2 = false;
		bool lm = false;
		bool lzcnt = false;
		bool xop = false;
		bool avx = false;
		bool avx2 = false;
		bool fma = false;
		bool avx512f = false;
		bool avx512bw = false;
		bool avx512vl = false;
		bool avx512dq = false;
		bool avx512ifma = false;
		bool avx512pf = false;
		bool avx512er = false;
		bool avx512cd = false;
		bool avx512vbmi = false;
		bool avx512vbmi2 = false;
		bool avx512vnni = false;
		bool avx512bitalg = false;
		bool avx512vpopcntdq = false;
		bool gfni = false;
		bool vaes = false;
		bool vpclmulqdq = false;
		bool avx_vnni = false;
		bool avx_ifma = false;
		bool sha = false;
		bool movdiri = false;
		bool movdir64b = false;
		bool serialize = false;
		bool tsxldtrk = false;
		bool pku = false;
		bool ospke = false;
		bool rdpid = false;
		bool waitpkg = false;
		bool clwb = false;
		bool clflushopt = false;
		bool sgx = false;
		bool avx512_4vnniw = false;
		bool avx512_4fmaps = false;
		bool avx512_vp2intersect = false;
		bool amx_tile = false;
		bool amx_int8 = false;
		bool amx_bf16 = false;
		bool amx_fp16 = false;
		bool ibt = false;
		bool shstk = false;
		bool fsrm = false;
		bool hreset = false;
		bool uintr = false;
		bool avx512_bf16 = false;
		bool avx512_fp16 = false;
		bool avx10 = false;
		uint32_t avx10_version = 0u;
		avx10_vector_length_t avx10_vector_length = avx10_vector_length_t::AVX_10_LENGTH_UNKNOWN;
		bool apx = false;
	};

	struct tsc_clock_info {
		bool is_invariant_tsc = false;

		uint64_t tsc_frequency_hz = 0ull;
		uint64_t crystal_clock_hz = 0ull;
	};

	struct cpu_brand_info {
		std::string brand_string;
		enum {
			INTEL,
			AMD,
			OTHER
		} vendor = INTEL;
		int32_t family = 0;
		int32_t model = 0;
	};

	[[nodiscard]] std::string get_cpu_brand_string() noexcept;
	[[nodiscard]] x86_extensions get_cpu_supported_isas() noexcept;
	[[nodiscard]] std::string get_cpu_isa_string() noexcept;
	[[nodiscard]] tsc_clock_info get_tsc_clock_info() noexcept;
	[[nodiscard]] cpu_brand_info get_cpu_brand_info() noexcept;

	[[nodiscard]] cpuid_result cpuid(uint32_t leaf, uint32_t subleaf) noexcept;
	void cpuid(uint32_t leaf, uint32_t subleaf, cpuid_result& out) noexcept;
	[[nodiscard]] uint64_t xgetbv(uint32_t index) noexcept;
} // namespace bench_box
#endif
