#pragma once

namespace bench_box {
	[[nodiscard]] timer_source detect_timer_source_x86(int64_t qpc_frequency_hz, int64_t tsc_frequency_hz, bool has_invariant_tsc) noexcept;
} // namespace bench_box
