#include <bench_box/isa/x86/cpuid.h>
#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
#include <array>
#include <string>
#include <cstring>

#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
	#include <cpuid.h>
#endif

namespace bench_box {
	[[nodiscard]] cpuid_result cpuid(uint32_t leaf, uint32_t subleaf) noexcept {
		cpuid_result result{};
		#if defined(_MSC_VER)
			__cpuidex(&result.eax, leaf, subleaf);
		#else
			__cpuid_count(leaf, subleaf, result.eax, result.ebx, result.ecx, result.edx);
		#endif
		return result;
	}

	void cpuid(uint32_t leaf, uint32_t subleaf, cpuid_result& out) noexcept {
		#if defined(_MSC_VER)
			__cpuidex(&out.eax, leaf, subleaf);
		#else
			__cpuid_count(leaf, subleaf, out.eax, out.ebx, out.ecx, out.edx);
		#endif
	}

	[[nodiscard]] uint64_t xgetbv(uint32_t index) noexcept {
		#if defined(_MSC_VER)
			return _xgetbv(index);
		#else
			uint32_t eax;
			uint32_t edx;
			__asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
			return (static_cast<uint64_t>(edx) << 32) | eax;
		#endif
	}


	[[nodiscard]] std::string get_cpu_brand_string() noexcept {
		std::array<char, 49> brand{};

		cpuid_result r = cpuid(0x80000000, 0);
		uint32_t max_ext_id = static_cast<uint32_t>(r.eax);

		if (max_ext_id >= 0x80000004) {
			r = cpuid(0x80000002, 0);
			std::memcpy(brand.data() + 0, &r.eax, 4);
			std::memcpy(brand.data() + 4, &r.ebx, 4);
			std::memcpy(brand.data() + 8, &r.ecx, 4);
			std::memcpy(brand.data() + 12, &r.edx, 4);

			r = cpuid(0x80000003, 0);
			std::memcpy(brand.data() + 16, &r.eax, 4);
			std::memcpy(brand.data() + 20, &r.ebx, 4);
			std::memcpy(brand.data() + 24, &r.ecx, 4);
			std::memcpy(brand.data() + 28, &r.edx, 4);

			r = cpuid(0x80000004, 0);
			std::memcpy(brand.data() + 32, &r.eax, 4);
			std::memcpy(brand.data() + 36, &r.ebx, 4);
			std::memcpy(brand.data() + 40, &r.ecx, 4);
			std::memcpy(brand.data() + 44, &r.edx, 4);
		}

		brand[48] = '\0';
		return std::string(brand.data());
	}


	[[nodiscard]] x86_extensions get_cpu_supported_isas() noexcept {
		x86_extensions result{};

		cpuid_result r = cpuid(0, 0);
		auto max_basic = r.eax;

		r = cpuid(0x80000000, 0);
		auto max_ext = r.eax;

		r = cpuid(1, 0);
		auto ecx = r.ecx;
		auto edx = r.edx;

		result.mmx = (edx & (1 << 23));
		result.sse = (edx & (1 << 25));
		result.sse2 = (edx & (1 << 26));
		result.sse3 = (ecx & (1 << 0));
		result.ssse3 = (ecx & (1 << 9));
		result.sse4_1 = (ecx & (1 << 19));
		result.sse4_2 = (ecx & (1 << 20));
		result.popcnt = (ecx & (1 << 23));
		result.aes = (ecx & (1 << 25));
		result.fma = (ecx & (1 << 12));
		result.rdrand = (ecx & (1 << 30));

		bool osxsave = (ecx & (1 << 27));
		bool avx_hw  = (ecx & (1 << 28));

		bool avx_enabled = false;
		bool avx512_os_enabled = false;

		if (osxsave && avx_hw) {
			uint64_t xcr0 = xgetbv(0);

			avx_enabled = ((xcr0 & 0x6) == 0x6);
			avx512_os_enabled = ((xcr0 & 0xE6) == 0xE6);

			result.avx = avx_enabled;
		}

		if (max_basic >= 7) {
			r = cpuid(7, 0);
			auto ebx = r.ebx;
			auto ecx7 = r.ecx;
			auto edx7 = r.edx;

			result.bmi1 = (ebx & (1 << 3));
			result.bmi2 = (ebx & (1 << 8));

			result.avx2 = ((ebx & (1 << 5)) && avx_enabled);

			result.avx512f = ((ebx & (1 << 16)) && avx512_os_enabled);
			result.avx512dq = ((ebx & (1 << 17)) && avx512_os_enabled);
			result.avx512ifma = ((ebx & (1 << 21)) && avx512_os_enabled);
			result.avx512pf = ((ebx & (1 << 26)) && avx512_os_enabled);
			result.avx512er = ((ebx & (1 << 27)) && avx512_os_enabled);
			result.avx512cd = ((ebx & (1 << 28)) && avx512_os_enabled);
			result.avx512bw = ((ebx & (1 << 30)) && avx512_os_enabled);
			result.avx512vl = ((ebx & (1 << 31)) && avx512_os_enabled);

			result.avx512vbmi = ((ecx7 & (1 << 1)) && avx512_os_enabled);
			result.gfni = ((ecx7 & (1 << 8)) && avx_enabled);
			result.vaes = ((ecx7 & (1 << 9)) && avx_enabled);
			result.vpclmulqdq = ((ecx7 & (1 << 10)) && avx_enabled);
			result.avx512vnni = ((ecx7 & (1 << 11)) && avx512_os_enabled);
			result.avx512bitalg = ((ecx7 & (1 << 12)) && avx512_os_enabled);
			result.avx512vpopcntdq = ((ecx7 & (1 << 14)) && avx512_os_enabled);
			result.rdpid = (ecx7 & (1 << 22));
			result.clflushopt = (ecx7 & (1 << 23));
			result.clwb = (ecx7 & (1 << 24));
			result.sha = (ecx7 & (1 << 29));
			result.avx512vbmi2 = ((ecx7 & (1 << 6)) && avx512_os_enabled);
			result.waitpkg = (ecx7 & (1 << 5));
			result.movdiri = (ecx7 & (1 << 27));
			result.movdir64b = (ecx7 & (1 << 28));
			result.tsxldtrk = (ecx7 & (1 << 16));
			result.pku = (ecx7 & (1 << 3));
			result.ospke = (ecx7 & (1 << 4));

			result.avx512_4vnniw = ((edx7 & (1 << 2)) && avx512_os_enabled);
			result.avx512_4fmaps = ((edx7 & (1 << 3)) && avx512_os_enabled);
			result.avx512_vp2intersect = ((edx7 & (1 << 8)) && avx512_os_enabled);
			result.amx_tile = (edx7 & (1 << 24));
			result.amx_int8 = (edx7 & (1 << 25));
			result.amx_bf16 = (edx7 & (1 << 22));
			result.amx_fp16 = (edx7 & (1 << 23));
			result.ibt = (edx7 & (1 << 20));
			result.shstk = (edx7 & (1 << 7));
			result.fsrm = (edx7 & (1 << 4));
			result.hreset = (edx7 & (1 << 22));
			result.uintr = (edx7 & (1 << 5));

			r = cpuid(7, 1);
			auto eax1 = r.eax;
			auto ecx1 = r.ecx;
			auto edx1 = r.edx;

			result.avx512_bf16 = ((eax1 & (1 << 5)) && avx512_os_enabled);
			result.avx512_fp16 = ((eax1 & (1 << 23)) && avx512_os_enabled);

			result.serialize = (ecx1 & (1 << 14));

			result.avx10 = (edx1 & (1 << 19)) && avx512_os_enabled;
			result.apx = (edx1 & (1 << 21));

			if (result.avx10 && max_basic >= 0x24) {
				r = cpuid(0x24, 0);
				result.avx10_version = (r.ebx & 0xFF);
				auto ebx24 = r.ebx;
				if (ebx24 & (1 << 18)) {
					result.avx10_vector_length = avx10_vector_length_t::AVX_10_512;
				} else if (ebx24 & (1 << 17)) {
					result.avx10_vector_length = avx10_vector_length_t::AVX_10_256;
				} else if (ebx24 & (1 << 16)) {
					result.avx10_vector_length = avx10_vector_length_t::AVX_10_128;
				} else {
					result.avx10_vector_length = avx10_vector_length_t::AVX_10_LENGTH_UNKNOWN;
				}
			}
		}

		if (max_ext >= static_cast<int32_t>(0x80000001)) {
			r = cpuid(0x80000001, 0);
			auto ecx8 = r.ecx;
			auto edx8 = r.edx;

			result.lm = (edx8 & (1 << 29));
			result.lzcnt = (ecx8 & (1 << 5));
			result.sse4a = (ecx8 & (1 << 6));
			result.xop = (ecx8 & (1 << 11));
		}

		return result;
	}

	std::string get_cpu_isa_string() noexcept {
		auto ext = get_cpu_supported_isas();
		std::string result;

		auto append = [&](bool supported, const char* name) {
			if (supported) {
				if (!result.empty()){
					result += ' ';
				}
				result += name;
			}
		};

		append(ext.mmx, "MMX");
		append(ext.sse, "SSE");
		append(ext.sse2, "SSE2");
		append(ext.sse3, "SSE3");
		append(ext.ssse3, "SSSE3");
		append(ext.sse4_1, "SSE4.1");
		append(ext.sse4_2, "SSE4.2");
		append(ext.sse4a, "SSE4a");
		append(ext.popcnt, "POPCNT");
		append(ext.aes, "AES");
		append(ext.fma, "FMA");
		append(ext.rdrand, "RDRAND");
		append(ext.bmi1, "BMI1");
		append(ext.bmi2, "BMI2");
		append(ext.lzcnt, "LZCNT");
		append(ext.xop, "XOP");
		append(ext.avx, "AVX");
		append(ext.avx2, "AVX2");
		if (ext.avx10) {
			if (!result.empty()) {
				result += ' ';
			}
			result += "AVX10";
			if (ext.avx10_version > 0) {
				result += '.';
				result += std::to_string(ext.avx10_version);
			}
			switch (ext.avx10_vector_length) {
				case avx10_vector_length_t::AVX_10_LENGTH_UNKNOWN:
					break;
				case avx10_vector_length_t::AVX_10_128:
					result += '-';
					result += "128";
					break;
				case avx10_vector_length_t::AVX_10_256:
					result += '-';
					result += "256";
					break;
				case avx10_vector_length_t::AVX_10_512:
					result += '-';
					result += "512";
					break;
			};
		} else {
			append(ext.avx512f, "AVX512F");
			append(ext.avx512bw, "AVX512BW");
			append(ext.avx512vl, "AVX512VL");
			append(ext.avx512dq, "AVX512DQ");
			append(ext.avx512ifma, "AVX512IFMA");
			append(ext.avx512pf, "AVX512PF");
			append(ext.avx512er, "AVX512ER");
			append(ext.avx512cd, "AVX512CD");
			append(ext.avx512vbmi, "AVX512VBMI");
			append(ext.avx512vbmi2, "AVX512VBMI2");
			append(ext.avx512vnni, "AVX512VNNI");
			append(ext.avx512bitalg, "AVX512BITALG");
			append(ext.avx512vpopcntdq, "AVX512VPOPCNTDQ");
			append(ext.avx512_bf16, "AVX512_BF16");
			append(ext.avx512_fp16, "AVX512_FP16");
			append(ext.avx512_4vnniw, "AVX512_4VNNIW");
			append(ext.avx512_4fmaps, "AVX512_4FMAPS");
			append(ext.avx512_vp2intersect, "AVX512_VP2INTERSECT");
		}
		append(ext.apx, "APX");
		append(ext.gfni, "GFNI");
		append(ext.vaes, "VAES");
		append(ext.vpclmulqdq, "VPCLMULQDQ");
		append(ext.sha, "SHA");
		append(ext.rdpid, "RDPID");
		append(ext.waitpkg, "WAITPKG");
		append(ext.movdiri, "MOVDIRI");
		append(ext.movdir64b, "MOVDIR64B");
		append(ext.serialize, "SERIALIZE");
		append(ext.tsxldtrk, "TSXLDTRK");
		append(ext.pku, "PKU");
		append(ext.ospke, "OSPKE");
		append(ext.sgx, "SGX");
		append(ext.clwb, "CLWB");
		append(ext.clflushopt, "CLFLUSHOPT");
		append(ext.amx_tile, "AMX_TILE");
		append(ext.amx_int8, "AMX_INT8");
		append(ext.amx_bf16, "AMX_BF16");
		append(ext.amx_fp16, "AMX_FP16");
		append(ext.ibt, "CET_IBT");
		append(ext.shstk, "CET_SHSTK");
		append(ext.fsrm, "FSRM");
		append(ext.hreset, "HRESET");
		append(ext.uintr, "UINTR");

		append(ext.apx, "APX");

		return result;
	}

	[[nodiscard]] tsc_clock_info get_tsc_clock_info() noexcept {
		tsc_clock_info result{};

		cpuid_result r = cpuid(0x80000007, 0);
		result.is_invariant_tsc = (r.edx & (1 << 8)) != 0;

		r = cpuid(0, 0);
		const auto max_basic = r.eax;

		uint64_t tsc_ratio_denominator = 0u;
		uint64_t tsc_ratio_numerator = 0u;

		uint64_t base_frequency_mhz = 0u;
		// not currently used, left here for ease if needed in the futrue
		[[maybe_unused]] uint64_t max_frequency_mhz = 0u;
		[[maybe_unused]] uint64_t bus_frequency_mhz = 0u;

		if (max_basic >= 0x15) {
			r = cpuid(0x15, 0);
			tsc_ratio_denominator = static_cast<uint64_t>(r.eax);
			tsc_ratio_numerator = static_cast<uint64_t>(r.ebx);
			result.crystal_clock_hz = static_cast<uint64_t>(r.ecx);
		}

		if (max_basic >= 0x16) {
			r = cpuid(0x16, 0);
			base_frequency_mhz = static_cast<uint64_t>(r.eax & 0xFFFF);
			max_frequency_mhz = static_cast<uint64_t>(r.ebx & 0xFFFF);
			bus_frequency_mhz = static_cast<uint64_t>(r.ecx & 0xFFFF);
		}

		if (result.crystal_clock_hz != 0u && tsc_ratio_numerator != 0u && tsc_ratio_denominator != 0u) {
			result.tsc_frequency_hz = result.crystal_clock_hz * tsc_ratio_numerator / tsc_ratio_denominator;
		} else {
			result.tsc_frequency_hz = base_frequency_mhz * 1'000ull;
		}

		return result;
	}

	[[nodiscard]] cpu_brand_info get_cpu_brand_info() noexcept {
		cpu_brand_info result{};
		result.brand_string = get_cpu_brand_string();

		if (result.brand_string.find("Intel") != std::string::npos) {
			result.vendor = cpu_brand_info::INTEL;
		} else if (result.brand_string.find("AMD") != std::string::npos) {
			result.vendor = cpu_brand_info::AMD;
		} else {
			result.vendor = cpu_brand_info::OTHER;
		}

		cpuid_result r = cpuid(1, 0);
		result.family = (r.eax >> 8) & 0x0F;
		result.model = ((r.eax >> 16) & 0x0F) << 4 | ((r.eax >> 4) & 0x0F);

		return result;
	}
} // namespace bench_box
#endif
