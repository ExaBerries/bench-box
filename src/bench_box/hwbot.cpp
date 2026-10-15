#include <bench_box/hwbot.h>

#include <bench_box/isa/x86/cpu_models.h>
#include <bench_box/isa/x86/cpuid.h>

#include <algorithm>

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#include <versionhelpers.h>
#endif

namespace {
	[[nodiscard]] bool is_windows_8_or_newer() noexcept {
		#if defined(_WIN32)
			return IsWindows8OrGreater();
		#else
			return false;
		#endif
	}
} // namespace

namespace bench_box {
	// implemented with the rules as of oct 15th, 2026
	[[nodiscard]] bool is_hwbot_compliant(const timer& t) noexcept {
		if (!is_windows_8_or_newer()) {
			return true;
		}
		#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
			const auto cpu_info = get_cpu_brand_info();
			if (cpu_info.vendor == cpu_brand_info::INTEL) {
				if (cpu_info.model >= INTEL_SKYLAKE_MOBILE_MODEL && cpu_info.model != INTEL_BROADWELL_E_MODEL) {
					return true;
				}
			}
		#endif
		return t.source == timer_source::HPET; // if not before win 8 and not skylake or later, hpet is required
	}
} // namespace bench_box