#pragma once
#include <cstdint>
#include <optional>
#include <string_view>

namespace bench_box {
	enum class thread_priority : uint8_t {
		background = 0u,
		low,
		normal, // os default
		high,
		realtime, // opt in only: needs cap_sys_nice / root / an elevated windows token, can starve the system
	};

	[[nodiscard]] const char* thread_priority_name(thread_priority prio) noexcept;
	[[nodiscard]] std::optional<thread_priority> parse_thread_priority(std::string_view str) noexcept;

	[[nodiscard]] bool disable_process_power_throttling() noexcept; // only works on windows

	// returns whether it actually applied
	// call from main(), after this any spawned thread inherits the setting
	// windows: sets the priority class, call before set_thread_priority
	[[nodiscard]] bool set_process_priority(thread_priority prio) noexcept;

	// returns whether the requested level was actually applied
	// call from within the thread
	[[nodiscard]] bool set_thread_priority(thread_priority prio) noexcept;
} // namespace bench_box
