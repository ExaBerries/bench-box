#include <gtest/gtest.h>
#include <bench_box/isa/isa_macro.h>
#include <bench_box/hwbot.h>
#include <bench_box/hwbot_int.h>

#if defined(BENCH_BOX_ISA_X86_64) || defined(BENCH_BOX_ISA_X86)
#include <bench_box/isa/x86/cpu_models.h>

using namespace bench_box;

TEST(Hwbot, PreWindows8AlwaysCompliant) {
	cpu_brand_info cpu{};
	cpu.vendor = cpu_brand_info::OTHER;
	EXPECT_TRUE(is_hwbot_compliant_x86(false, cpu, false));
	EXPECT_TRUE(is_hwbot_compliant_x86(true, cpu, false));
}

TEST(Hwbot, IntelSkylakeExempt) {
	cpu_brand_info cpu{};
	cpu.vendor = cpu_brand_info::INTEL;
	cpu.model = INTEL_SKYLAKE_MOBILE_MODEL;
	EXPECT_TRUE(is_hwbot_compliant_x86(false, cpu, true));
	EXPECT_TRUE(is_hwbot_compliant_x86(true, cpu, true));
}

TEST(Hwbot, IntelNewerExempt) {
	cpu_brand_info cpu{};
	cpu.vendor = cpu_brand_info::INTEL;
	cpu.model = INTEL_ALDER_LAKE_DESKTOP_MODEL;
	EXPECT_TRUE(is_hwbot_compliant_x86(false, cpu, true));
	EXPECT_TRUE(is_hwbot_compliant_x86(true, cpu, true));
}

TEST(Hwbot, OlderRequiresHpet) {
	cpu_brand_info cpu{};
	cpu.vendor = cpu_brand_info::INTEL;
	cpu.model = INTEL_SANDY_BRIDGE_E_MODEL;
	EXPECT_FALSE(is_hwbot_compliant_x86(false, cpu, true));
	EXPECT_TRUE(is_hwbot_compliant_x86(true, cpu, true));
}

TEST(Hwbot, AmdRequiresHpet) {
	cpu_brand_info cpu{};
	cpu.vendor = cpu_brand_info::AMD;
	cpu.model = AMD_ZEN4_MODEL;
	EXPECT_FALSE(is_hwbot_compliant_x86(false, cpu, true));
	EXPECT_TRUE(is_hwbot_compliant_x86(true, cpu, true));
}
#endif