#include <bench_box/hwbot.h>
#include <bench_box/hwbot_int.h>

#include <bench_box/isa/x86/cpu_models.h>
#include <bench_box/isa/x86/cpuid.h>

#include <algorithm>

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#include <versionhelpers.h>
#endif

namespace bench_box {
	// implemented with the rules as of oct 15th, 2026
	#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
		[[nodiscard]] bool is_hwbot_compliant_x86(bool is_hpet, const cpu_brand_info& cpu_info, bool is_win8_or_later) noexcept {
			if (!is_win8_or_later) {
				return true;
			}
			if (cpu_info.vendor == cpu_brand_info::INTEL) {
				if (is_intel_skylake_or_newer(cpu_info.family, cpu_info.model)) {
					return true;
				}
			}
			return is_hpet; // if not before win 8 and not skylake or later, hpet is required
		}
	#endif

	[[nodiscard]] bool is_hwbot_compliant([[maybe_unused]] const timer& t) noexcept {
		#if defined(_WIN32)
			#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
				const auto cpu_info = get_cpu_brand_info();
				return is_hwbot_compliant_x86(t.source == timer_source::HPET, cpu_info, IsWindows8OrGreater());
			#endif
		#endif
		return false;
	}
} // namespace bench_box