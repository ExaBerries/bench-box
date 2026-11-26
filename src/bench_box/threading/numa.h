#pragma once
#include <bench_box/threading/threading.h>
#include <bench_box/threading/topology.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace bench_box {
	[[nodiscard]] std::optional<uint32_t> numa_node_of_processor_loc(const topology_tree& tree, processor_location location) noexcept;
	[[nodiscard]] std::optional<uint32_t> numa_node_of_current_thread(const topology_tree& tree) noexcept;

	struct numa_location {
		processor_location processor{};
		uint32_t node = 0u;
	};

	class numa_thread_registry {
	private:
		const topology_tree& topology;
		std::vector<std::optional<numa_location>> locations;
	public:
		numa_thread_registry(const topology_tree& topo_tree, uint32_t thread_count);
		numa_thread_registry(topology_tree&&) = delete;

		numa_thread_registry(const numa_thread_registry&) = delete;
		numa_thread_registry& operator=(const numa_thread_registry&) = delete;
		numa_thread_registry(numa_thread_registry&&) = delete;
		numa_thread_registry& operator=(numa_thread_registry&&) = delete;

		void register_thread(uint32_t thread_id) noexcept;
		void register_thread(uint32_t thread_id, processor_location location) noexcept;

		[[nodiscard]] uint32_t thread_count() const noexcept;
		[[nodiscard]] std::optional<numa_location> numa_location_of(uint32_t thread_id) const noexcept;
		[[nodiscard]] std::optional<uint32_t> node_of(uint32_t thread_id) const noexcept;
		[[nodiscard]] bool all_registered() const noexcept;
	};
} // namespace bench_box
