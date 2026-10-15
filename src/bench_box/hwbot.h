#pragma once
#include <bench_box/timer.h>

namespace bench_box {
	[[nodiscard]] bool is_hwbot_compliant(const timer& t) noexcept;
} // namespace bench_box