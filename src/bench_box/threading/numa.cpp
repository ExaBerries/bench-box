#include <bench_box/threading/numa.h>
#include <algorithm>

namespace bench_box {
	[[nodiscard]] std::optional<uint32_t> numa_node_of_processor_loc(const topology_tree& tree, processor_location location) noexcept {
		for (const auto& node : tree.numa_nodes) {
			for (const auto& llc : node.llc_groups) {
				for (const auto& core : llc.cores) {
					for (const auto& lp : core.threads) {
						if (lp.group_id == location.group_id && lp.os_id == location.os_id) {
							return node.node_id;
						}
					}
				}
			}
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<uint32_t> numa_node_of_current_thread(const topology_tree& tree) noexcept {
		const auto location = current_processor();
		if (!location) {
			return std::nullopt;
		}
		return numa_node_of_processor_loc(tree, *location);
	}

	numa_thread_registry::numa_thread_registry(const topology_tree& topo_tree, uint32_t thread_count) : topology(topo_tree), locations(thread_count) {
	}

	void numa_thread_registry::register_thread(uint32_t thread_id) noexcept {
		const auto location = current_processor();
		if (!location) {
			return;
		}
		register_thread(thread_id, *location);
	}

	void numa_thread_registry::register_thread(uint32_t thread_id, processor_location location) noexcept {
		if (thread_id >= locations.size()) {
			return;
		}

		const auto node = numa_node_of_processor_loc(topology, location);
		if (!node) {
			return;
		}

		locations[thread_id] = numa_location{location, *node};
	}

	[[nodiscard]] uint32_t numa_thread_registry::thread_count() const noexcept {
		return static_cast<uint32_t>(locations.size());
	}

	[[nodiscard]] std::optional<numa_location> numa_thread_registry::numa_location_of(uint32_t thread_id) const noexcept {
		if (thread_id >= locations.size()) {
			return std::nullopt;
		}
		return locations[thread_id];
	}

	[[nodiscard]] std::optional<uint32_t> numa_thread_registry::node_of(uint32_t thread_id) const noexcept {
		if (thread_id >= locations.size()) {
			return std::nullopt;
		}
		const auto& entry = locations[thread_id];
		if (!entry) {
			return std::nullopt;
		}
		return entry->node;
	}

	[[nodiscard]] bool numa_thread_registry::all_registered() const noexcept {
		return std::all_of(
			locations.begin(),
			locations.end(),
			[](const std::optional<numa_location>& entry) noexcept {
				return entry.has_value();
			}
		);
	}
} // namespace bench_box
