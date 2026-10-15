#pragma once

#include <array>
#include <cstdint>

namespace bench_box {
	// Intel Core (family 6) model IDs for quick uarch identification.
	inline constexpr int32_t INTEL_CORE2_MEROM_MODEL = 0x0F; // Merom/Conroe
	inline constexpr int32_t INTEL_CORE2_PENRYN_MODEL = 0x17; // Penryn/Wolfdale
	inline constexpr int32_t INTEL_CORE2_DUNNINGTON_MODEL = 0x1D; // Dunnington

	inline constexpr int32_t INTEL_NEHALEM_BLOOMFIELD_MODEL = 0x1A; // Bloomfield
	inline constexpr int32_t INTEL_NEHALEM_LYNNFIELD_MODEL = 0x1E; // Lynnfield
	inline constexpr int32_t INTEL_NEHALEM_CLARKSFIELD_MODEL = 0x1F; // Clarksfield
	inline constexpr int32_t INTEL_NEHALEM_GULFTOWN_MODEL = 0x2E; // Gulftown

	inline constexpr int32_t INTEL_WESTMERE_CLARKDALE_MODEL = 0x25; // Clarkdale/Arrandale
	inline constexpr int32_t INTEL_WESTMERE_EP_MODEL = 0x2C; // Westmere-EP
	inline constexpr int32_t INTEL_WESTMERE_EX_MODEL = 0x2F; // Westmere-EX

	inline constexpr int32_t INTEL_SANDY_BRIDGE_MODEL = 0x2A; // Sandy Bridge (mobile)
	inline constexpr int32_t INTEL_SANDY_BRIDGE_E_MODEL = 0x2D; // Sandy Bridge-E

	inline constexpr int32_t INTEL_IVY_BRIDGE_MODEL = 0x3A; // Ivy Bridge
	inline constexpr int32_t INTEL_IVY_BRIDGE_E_MODEL = 0x3E; // Ivy Bridge-E

	inline constexpr int32_t INTEL_HASWELL_DESKTOP_MODEL = 0x3C; // Haswell (desktop)
	inline constexpr int32_t INTEL_HASWELL_MOBILE_MODEL_A = 0x45; // Haswell (mobile)
	inline constexpr int32_t INTEL_HASWELL_MOBILE_MODEL_B = 0x46; // Haswell (mobile)
	inline constexpr int32_t INTEL_HASWELL_E_MODEL = 0x3F; // Haswell-E

	inline constexpr int32_t INTEL_BROADWELL_MODEL = 0x3D; // Broadwell
	inline constexpr int32_t INTEL_BROADWELL_H_MODEL = 0x47; // Broadwell-H
	inline constexpr int32_t INTEL_BROADWELL_E_MODEL = 0x4F; // Broadwell-E

	inline constexpr int32_t INTEL_SKYLAKE_MOBILE_MODEL = 0x4E; // Skylake (mobile U/Y)
	inline constexpr int32_t INTEL_SKYLAKE_DESKTOP_MODEL = 0x5E; // Skylake (desktop S/H)
	inline constexpr int32_t INTEL_SKYLAKE_X_MODEL = 0x55; // Skylake-X/SP

	inline constexpr int32_t INTEL_KABY_LAKE_MOBILE_MODEL = 0x8E; // Kaby Lake (mobile)
	inline constexpr int32_t INTEL_KABY_LAKE_DESKTOP_MODEL = 0x9E; // Kaby Lake (desktop)
	inline constexpr int32_t INTEL_COFFEE_LAKE_MODEL = 0x9E; // Coffee Lake (shares ID)

	inline constexpr int32_t INTEL_CANNON_LAKE_MODEL = 0x66; // Cannon Lake
	inline constexpr int32_t INTEL_ICE_LAKE_MOBILE_MODEL_A = 0x7D; // Ice Lake (mobile)
	inline constexpr int32_t INTEL_ICE_LAKE_MOBILE_MODEL_B = 0x7E; // Ice Lake (mobile)
	inline constexpr int32_t INTEL_TIGER_LAKE_MOBILE_MODEL_A = 0x8C; // Tiger Lake (mobile)
	inline constexpr int32_t INTEL_TIGER_LAKE_MOBILE_MODEL_B = 0x8D; // Tiger Lake (mobile)
	inline constexpr int32_t INTEL_ALDER_LAKE_DESKTOP_MODEL = 0x97; // Alder Lake (desktop)
	inline constexpr int32_t INTEL_ALDER_LAKE_MOBILE_MODEL = 0x9A; // Alder Lake (mobile)

	// AMD Zen family/model samples (family 17h/19h).
	inline constexpr int32_t AMD_ZEN_MODEL = 0x01; // Zen
	inline constexpr int32_t AMD_ZEN_PLUS_MODEL = 0x08; // Zen+
	inline constexpr int32_t AMD_ZEN2_MODEL = 0x31; // Zen 2
	inline constexpr int32_t AMD_ZEN3_MODEL = 0x21; // Zen 3
	inline constexpr int32_t AMD_ZEN4_MODEL = 0x61; // Zen 4
} // namespace bench_box