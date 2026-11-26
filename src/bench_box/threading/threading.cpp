#include <bench_box/threading/threading.h>
#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <windows.h>
#elif defined(__linux__)
	#include <pthread.h>
	#include <sched.h>
#endif

namespace bench_box {
	[[nodiscard]] std::optional<processor_location> current_processor() noexcept {
		#if defined(_WIN32)
			PROCESSOR_NUMBER number = {};
			GetCurrentProcessorNumberEx(&number);
			return processor_location{static_cast<uint32_t>(number.Group), static_cast<uint32_t>(number.Number)};
		#elif defined(__linux__)
			const auto cpu = sched_getcpu();
			if (cpu < 0) {
				return std::nullopt;
			}
			return processor_location{0u, static_cast<uint32_t>(cpu)};
		#else
			return std::nullopt;
		#endif
	}

	void assign_thread(processor_location location) noexcept {
		#if defined(_WIN32)
			GROUP_AFFINITY affinity = {};
			affinity.Mask = (1ULL << location.os_id);
			affinity.Group = static_cast<WORD>(location.group_id);
			SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
		#elif defined(__linux__)
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(location.os_id, &cpuset);
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
		#endif
	}

	void assign_thread(const affinity_map& aff_map, uint32_t thread_id) noexcept {
		if (thread_id < aff_map.size()) {
			assign_thread(aff_map[thread_id]);
		}
	}

	[[nodiscard]] affinity_map create_mapping_basic() noexcept {
		return {};
	}

	struct groups_t {
		std::vector<std::vector<const core*>> groups{};
		std::size_t max_threads = 0u;
	};

	static groups_t build_groups_int_numa_int_llc(const topology_tree& topo_tree, uint32_t perf_level) noexcept {
		std::vector<std::vector<const core*>> groups;
		std::size_t max_threads = 0u;
		std::size_t max_llc_count = 0u;
		for (const auto& node : topo_tree.numa_nodes) {
			max_llc_count = std::max(max_llc_count, node.llc_groups.size());
		}

		for (auto llc_index = 0u; llc_index < max_llc_count; llc_index++) {
			for (const auto& node : topo_tree.numa_nodes) {
				if (llc_index < node.llc_groups.size()) {
					std::vector<const core*> cores_in_group;
					for (const auto& c : node.llc_groups[llc_index].cores) {
						if (c.perf_level == perf_level && !c.threads.empty()) {
							cores_in_group.push_back(&c);
							max_threads = std::max(max_threads, c.threads.size());
						}
					}
					if (!cores_in_group.empty()) {
						groups.push_back(std::move(cores_in_group));
					}
				}
			}
		}
		return {std::move(groups), max_threads};
	}

	using thread_lists_t = std::vector<std::vector<const logical_processor*>>;

	static void initalize_thread_lists(thread_lists_t& thread_lists, std::size_t max_threads) noexcept {
		thread_lists.clear();
		for (auto i = 0u; i < max_threads; i++) {
			thread_lists.emplace_back(std::vector<const logical_processor*>{});
		}
	}

	static void interleave(thread_lists_t& thread_lists, const std::vector<std::vector<const core*>>& groups) noexcept {
		const auto max_threads = thread_lists.size();
		size_t max_cores = 0;
		for (const auto& g : groups) {
			max_cores = std::max(max_cores, g.size());
		}

		for (size_t i = 0; i < max_cores; i++) {
			for (const auto& g : groups) {
				if (i < g.size()) {
					for (auto t = 0u; t < max_threads; t++) {
						if (t < g[i]->threads.size()) {
							thread_lists[t].push_back(&g[i]->threads[t]);
						}
					}
				}
			}
		}
	}

	static void add_threads_to_result(affinity_map& result, const thread_lists_t& thread_lists) noexcept {
		for (const auto& thread_list : thread_lists) {
			for (const auto* lp : thread_list) {
				result.push_back({
					lp->group_id,
					lp->os_id
				});
			}
		}
	}

	[[nodiscard]] affinity_map create_mapping_numa(const topology_tree& topo_tree) noexcept {
		affinity_map result{};
		if (topo_tree.numa_nodes.empty()) {
			return result;
		}

		std::vector<groups_t> levels;
		levels.reserve(topo_tree.perf_level_count);
		std::size_t max_threads = 0u;
		for (auto lvl = 0u; lvl < topo_tree.perf_level_count; lvl++) {
			levels.push_back(build_groups_int_numa_int_llc(topo_tree, lvl));
			max_threads = std::max(max_threads, levels.back().max_threads);
		}

		std::vector<std::vector<const logical_processor*>> thread_lists;
		initalize_thread_lists(thread_lists, max_threads);
		for (const auto& level : levels) {
			interleave(thread_lists, level.groups);
		}
		add_threads_to_result(result, thread_lists);

		return result;
	}

	[[nodiscard]] affinity_map create_mapping_phte(const topology_tree& topo_tree) noexcept {
		affinity_map result{};
		if (topo_tree.numa_nodes.empty()) {
			return result;
		}

		std::vector<std::vector<const logical_processor*>> thread_lists;

		// each perf tier is fully separated (no interleaving across tiers)
		for (auto lvl = 0u; lvl < topo_tree.perf_level_count; lvl++) {
			auto [groups, mt] = build_groups_int_numa_int_llc(topo_tree, lvl);
			initalize_thread_lists(thread_lists, mt);
			interleave(thread_lists, groups);
			add_threads_to_result(result, thread_lists);
		}

		return result;
	}

	[[nodiscard]] uint32_t get_physical_core_count(const topology_tree& tree) noexcept {
		if (tree.numa_nodes.empty()) {
			return std::thread::hardware_concurrency(); // yes this will include hyperthreads but its the best we have...
		}

		uint32_t count = 0;
		for (const auto& node : tree.numa_nodes) {
			for (const auto& llc : node.llc_groups) {
				count += static_cast<uint32_t>(llc.cores.size());
			}
		}
		return count;
	}

} // namespace bench_box
