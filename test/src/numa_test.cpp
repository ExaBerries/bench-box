#include <gtest/gtest.h>
#include <bench_box/threading/numa.h>
#include "mapping.h"

namespace bench_box {
	TEST(Numa, NodeOfProcessor) {
		const auto tree = create_2s2c2t_tree();

		const auto node0 = numa_node_of_processor_loc(tree, processor_location{0u, 0u});
		ASSERT_TRUE(node0.has_value());
		EXPECT_EQ(*node0, 0u);

		const auto node1 = numa_node_of_processor_loc(tree, processor_location{0u, 4u});
		ASSERT_TRUE(node1.has_value());
		EXPECT_EQ(*node1, 1u);

		EXPECT_FALSE(numa_node_of_processor_loc(tree, processor_location{0u, 8u}).has_value());
	}

	TEST(NumaThreadRegistry, RecordsObservedNodes) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 4u);

		registry.register_thread(0u, processor_location{0u, 0u}); // node 0
		registry.register_thread(1u, processor_location{0u, 4u}); // node 1
		registry.register_thread(2u, processor_location{0u, 2u}); // node 0
		registry.register_thread(3u, processor_location{0u, 6u}); // node 1

		ASSERT_TRUE(registry.all_registered());
		EXPECT_EQ(registry.thread_count(), 4u);

		const auto node0 = registry.node_of(0u);
		ASSERT_TRUE(node0.has_value());
		EXPECT_EQ(*node0, 0u);
		const auto node1 = registry.node_of(1u);
		ASSERT_TRUE(node1.has_value());
		EXPECT_EQ(*node1, 1u);

		const auto location = registry.numa_location_of(2u);
		ASSERT_TRUE(location.has_value());
		EXPECT_EQ(location->node, 0u);
		EXPECT_EQ(location->processor.os_id, 2u);
	}

	TEST(NumaThreadRegistry, IncompleteRegistration) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 4u);

		registry.register_thread(0u, processor_location{0u, 0u});

		EXPECT_FALSE(registry.all_registered());
		EXPECT_TRUE(registry.node_of(0u).has_value());
		EXPECT_FALSE(registry.node_of(1u).has_value());
	}

	TEST(NumaThreadRegistry, ReRegistrationMovesThread) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 2u);

		registry.register_thread(0u, processor_location{0u, 0u}); // node 0
		registry.register_thread(1u, processor_location{0u, 2u}); // node 0

		const auto before = registry.node_of(0u);
		ASSERT_TRUE(before.has_value());
		EXPECT_EQ(*before, 0u);

		registry.register_thread(0u, processor_location{0u, 4u}); // node 1

		const auto after = registry.node_of(0u);
		ASSERT_TRUE(after.has_value());
		EXPECT_EQ(*after, 1u);

		const auto location = registry.numa_location_of(0u);
		ASSERT_TRUE(location.has_value());
		EXPECT_EQ(location->node, 1u);
		EXPECT_EQ(location->processor.os_id, 4u);

		EXPECT_EQ(registry.node_of(1u), 0u);
	}

	TEST(NumaThreadRegistry, UnresolvableLocationIsIgnored) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 2u);

		registry.register_thread(0u, processor_location{0u, 99u});

		EXPECT_FALSE(registry.all_registered());
		EXPECT_FALSE(registry.node_of(0u).has_value());
		EXPECT_FALSE(registry.numa_location_of(0u).has_value());

		registry.register_thread(0u, processor_location{0u, 0u});
		const auto node = registry.node_of(0u);
		ASSERT_TRUE(node.has_value());
		EXPECT_EQ(*node, 0u);
	}

	TEST(NumaThreadRegistry, OutOfRangeThreadIsIgnored) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 2u);

		registry.register_thread(5u, processor_location{0u, 0u});

		EXPECT_FALSE(registry.all_registered());
		EXPECT_FALSE(registry.node_of(5u).has_value());
	}

	TEST(NumaThreadRegistry, ThreadsSharingAProcessorStayInOneNode) {
		const auto tree = create_2s2c2t_tree();
		numa_thread_registry registry(tree, 3u);

		registry.register_thread(0u, processor_location{0u, 0u});
		registry.register_thread(1u, processor_location{0u, 0u});
		registry.register_thread(2u, processor_location{0u, 4u});

		ASSERT_TRUE(registry.all_registered());
		EXPECT_EQ(registry.node_of(0u), 0u);
		EXPECT_EQ(registry.node_of(1u), 0u);
		EXPECT_EQ(registry.node_of(2u), 1u);
	}
} // namespace bench_box
