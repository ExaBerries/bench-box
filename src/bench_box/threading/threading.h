#pragma once
#include <bench_box/threading/topology.h>
#include <cstdint>
#include <optional>
#include <string_view>
#include <thread>

namespace bench_box {
	struct threading_manager {
		const uint32_t thread_count = 0u;
		std::vector<logical_processor> thread_mapping{};
	};

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

	// call from within thread
	void assign_thread(uint32_t os_id, uint32_t group_id) noexcept;
	void assign_thread(const threading_manager& thread_manager, uint32_t thread_id) noexcept;

	[[nodiscard]] std::vector<logical_processor> create_mapping_basic() noexcept;
	[[nodiscard]] std::vector<logical_processor> create_mapping_numa(const topology_tree& topo_tree) noexcept;
	[[nodiscard]] std::vector<logical_processor> create_mapping_phte(const topology_tree& topo_tree) noexcept;

	[[nodiscard]] uint32_t get_default_thread_count(const topology_tree& tree) noexcept;
} // namespace bench_box
