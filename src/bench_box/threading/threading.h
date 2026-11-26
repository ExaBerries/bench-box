#pragma once
#include <bench_box/threading/priority.h>
#include <bench_box/threading/topology.h>
#include <cstdint>
#include <optional>
#include <string_view>
#include <thread>

namespace bench_box {
	// operating system notion of the logical processor for thread affinity
	struct processor_location {
		uint32_t group_id = 0u;
		uint32_t os_id = 0u;
	};

	// OS notion of logical processor the calling thread is running on right now
	[[nodiscard]] std::optional<processor_location> current_processor() noexcept;

	using affinity_map = std::vector<processor_location>;

	// call from within thread
	void assign_thread(processor_location location) noexcept;
	void assign_thread(const affinity_map& aff_map, uint32_t thread_id) noexcept;

	[[nodiscard]] affinity_map create_mapping_basic() noexcept;
	[[nodiscard]] affinity_map create_mapping_numa(const topology_tree& topo_tree) noexcept;
	[[nodiscard]] affinity_map create_mapping_phte(const topology_tree& topo_tree) noexcept;

	[[nodiscard]] uint32_t get_physical_core_count(const topology_tree& tree) noexcept;
} // namespace bench_box
