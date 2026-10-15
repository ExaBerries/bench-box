#pragma once
#include <bench_box/isa/isa_macro.h>
#include <bench_box/timer.h>
#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
#include <bench_box/isa/x86/cpuid.h>

namespace bench_box {
	[[nodiscard]] bool is_hwbot_compliant_x86(bool is_hpet, const cpu_brand_info& cpu_info, bool is_win8_or_later) noexcept;
} // namespace bench_box

#endif