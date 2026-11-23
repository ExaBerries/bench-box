#include <gtest/gtest.h>
#include <bench_box/memory/unique_array.h>

#include "memory_test_helpers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

using namespace bench_box;

static_assert(std::is_same_v<uarray<int>, unique_array<int>>);
static_assert(std::is_same_v<typename unique_array<int>::AllocatorType, std::allocator<int>>);
static_assert(std::is_same_v<typename uarray<int>::AllocatorType, std::allocator<int>>);
static_assert(!std::is_copy_constructible_v<unique_array<int>>);
static_assert(!std::is_copy_assignable_v<unique_array<int>>);
static_assert(std::is_nothrow_move_constructible_v<unique_array<int>>);
static_assert(std::is_nothrow_move_assignable_v<unique_array<int>>);
static_assert(std::is_nothrow_destructible_v<unique_array<int>>);
// no pointer/view construction
static_assert(!std::is_constructible_v<unique_array<int>, std::size_t, int*>);

// move noexcept follows the allocator
using throwing_move_array = unique_array<int, bb_test::throwing_move_allocator>;
static_assert(!std::is_nothrow_move_constructible_v<throwing_move_array>);
static_assert(!std::is_nothrow_move_assignable_v<throwing_move_array>);

TEST(UniqueArray, EmptyBeginEqualsEnd) {
	unique_array<int> empty;
	unique_array<int> zero(0);
	EXPECT_EQ(empty.begin(), empty.end());
	EXPECT_EQ(zero.begin(), zero.end());
	EXPECT_FALSE(empty.valid());
	EXPECT_TRUE(empty.empty());
	EXPECT_FALSE(empty.operator bool());
	EXPECT_TRUE(zero.empty());
	EXPECT_FALSE(zero.operator bool());
}

TEST(UniqueArray, AllocatesAndExposesSpan) {
	unique_array<std::uint64_t> data(64);
	ASSERT_TRUE(data.valid());
	ASSERT_EQ(data.size(), 64u);
	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.operator bool());

	auto values = data.span();
	ASSERT_EQ(values.size(), 64u);
	for (std::size_t i = 0; i < values.size(); i++) {
		values[i] = i;
	}
	EXPECT_EQ(values[63], 63u);
	EXPECT_EQ(data.data()[10], 10u);
}

TEST(UniqueArray, EmptyWhenAllocationFails) {
	unique_array<std::uint64_t> data(std::size_t{1} << 60);
	EXPECT_FALSE(data.valid());
	EXPECT_EQ(data.data(), nullptr);
	EXPECT_EQ(data.size(), 0u);
	EXPECT_TRUE(data.empty());
	EXPECT_FALSE(data.operator bool());
	EXPECT_TRUE(data.span().empty());
}

TEST(UniqueArray, MoveTransfersOwnership) {
	unique_array<std::uint64_t> first(16);
	ASSERT_TRUE(first.valid());
	auto* const ptr = first.data();

	unique_array<std::uint64_t> second(std::move(first));
	EXPECT_FALSE(first.valid());
	EXPECT_EQ(first.size(), 0u);
	EXPECT_TRUE(first.empty());
	EXPECT_FALSE(first.operator bool());
	EXPECT_TRUE(second.valid());
	EXPECT_FALSE(second.empty());
	EXPECT_TRUE(second.operator bool());
	EXPECT_EQ(second.data(), ptr);
}

TEST(UniqueArray, FailedElementConstructionDeallocates) {
	bb_test::allocation_counts counts;
	unique_array<bb_test::throwing_default_ctor, bb_test::counting_allocator<bb_test::throwing_default_ctor>> arr(
		4,
		bb_test::counting_allocator<bb_test::throwing_default_ctor>{counts}
	);

	EXPECT_EQ(counts.allocations, 1u);
	EXPECT_EQ(counts.deallocations, 1u);
	EXPECT_FALSE(arr.valid());
	EXPECT_TRUE(arr.empty());
	EXPECT_FALSE(arr.operator bool());
}

TEST(UniqueArray, ZeroSizeAllocationOwnsButIsEmpty) {
	unique_array<int, bb_test::nonnull_zero_allocator<int>> zero(0, bb_test::nonnull_zero_allocator<int>{});

	EXPECT_TRUE(zero.valid());
	EXPECT_TRUE(zero.empty());
	EXPECT_FALSE(zero.operator bool());
	EXPECT_NE(zero.data(), nullptr);
}

TEST(UniqueArray, MoveAssignmentFreesPreviousAllocation) {
	unique_array<std::uint64_t> first(16);
	unique_array<std::uint64_t> second(32);
	ASSERT_TRUE(first.valid());
	ASSERT_TRUE(second.valid());

	second = std::move(first);
	EXPECT_TRUE(second.valid());
	EXPECT_EQ(second.size(), 16u);
	EXPECT_NE(second.data(), nullptr);
	EXPECT_EQ(first.size(), 0u);
	EXPECT_TRUE(first.empty());
	EXPECT_FALSE(first.operator bool());
}

TEST(UniqueArray, SelfMoveAssignmentIsSafe) {
	unique_array<std::uint64_t> data(16);
	ASSERT_TRUE(data.valid());
	auto* const ptr = data.data();

	unique_array<std::uint64_t>& alias = data;
	data = std::move(alias);
	EXPECT_TRUE(data.valid());
	EXPECT_EQ(data.data(), ptr);
	EXPECT_EQ(data.size(), 16u);
}

TEST(UniqueArray, UArrayAliasWorks) {
	uarray<std::uint32_t> data(4);
	ASSERT_TRUE(data.valid());
	ASSERT_EQ(data.size(), 4u);
	data.data()[0] = 42u;
	EXPECT_EQ(data.span()[0], 42u);
}
