#include <gtest/gtest.h>
#include <bench_box/memory/memory.h>
#include <bench_box/memory/buffer.h>

#include "memory_test_helpers.h"

#include <cstddef>
#include <type_traits>
#include <utility>

using namespace bench_box;

// no non-owning view construction: buffer is always an owner
static_assert(!std::is_constructible_v<buffer<int>, std::size_t, int*>);
static_assert(!std::is_constructible_v<buffer<int, lp_flex_allocator_t<int>>, std::size_t, int*>);
static_assert(std::is_nothrow_move_constructible_v<buffer<int>>);
static_assert(std::is_nothrow_move_assignable_v<buffer<int>>);
static_assert(std::is_nothrow_destructible_v<buffer<int>>);

TEST(Buffer, DefaultAlloc) {
	using namespace bench_box;

	buffer<uint32_t> buf(64);
	buf.free();
}

TEST(Buffer, EmptyBeginEqualsEnd) {
	using namespace bench_box;

	buffer<int> empty;
	buffer<int> zero(std::size_t{0});
	EXPECT_EQ(empty.begin(), empty.end());
	EXPECT_EQ(zero.begin(), zero.end());
	EXPECT_FALSE(empty.valid());
	EXPECT_TRUE(empty.empty());
	EXPECT_FALSE(empty.operator bool());
	EXPECT_TRUE(zero.empty());
	EXPECT_FALSE(zero.operator bool());
	zero.free();
}

TEST(Buffer, ExposesSpan) {
	using namespace bench_box;

	buffer<uint32_t> buf(8);
	ASSERT_TRUE(buf.valid());
	EXPECT_FALSE(buf.empty());
	EXPECT_TRUE(buf.operator bool());

	auto values = buf.span();
	ASSERT_EQ(values.size(), buf.size);
	for (std::size_t i = 0; i < values.size(); i++) {
		values[i] = static_cast<uint32_t>(i);
	}

	const buffer<uint32_t>& view = buf;
	auto const_values = view.span();
	ASSERT_EQ(const_values.size(), values.size());
	EXPECT_EQ(const_values[7], 7u);

	buf.free();
	EXPECT_FALSE(buf.valid());
	EXPECT_TRUE(buf.empty());
	EXPECT_FALSE(buf.operator bool());
	EXPECT_TRUE(buf.span().empty());
}

TEST(Buffer, MoveConstructionTransfersOwnership) {
	using namespace bench_box;

	buffer<int> first(16);
	ASSERT_TRUE(first.valid());
	auto* const ptr = first.data;

	buffer<int> second(std::move(first));
	EXPECT_FALSE(first.valid());
	EXPECT_TRUE(first.empty());
	EXPECT_EQ(first.size, 0u);
	EXPECT_TRUE(second.valid());
	EXPECT_FALSE(second.empty());
	EXPECT_TRUE(second.operator bool());
	EXPECT_EQ(second.data, ptr);

	first.free(); // moved-from free is a no-op
	second.free();
}

TEST(Buffer, FailedElementConstructionDeallocates) {
	using namespace bench_box;

	bb_test::allocation_counts counts;
	buffer<bb_test::throwing_default_ctor, bb_test::counting_allocator<bb_test::throwing_default_ctor>> buf(
		4,
		bb_test::counting_allocator<bb_test::throwing_default_ctor>{counts}
	);

	EXPECT_EQ(counts.allocations, 1u);
	EXPECT_EQ(counts.deallocations, 1u);
	EXPECT_FALSE(buf.valid());
	EXPECT_TRUE(buf.empty());
	EXPECT_FALSE(buf.operator bool());
}

TEST(Buffer, ZeroSizeAllocationOwnsButIsEmpty) {
	using namespace bench_box;

	buffer<int, bb_test::nonnull_zero_allocator<int>> zero(std::size_t{0}, bb_test::nonnull_zero_allocator<int>{});

	EXPECT_TRUE(zero.valid());
	EXPECT_TRUE(zero.empty());
	EXPECT_FALSE(zero.operator bool());
	// boolean contexts on buffer go through operator TYPE*, not operator bool
	EXPECT_NE(zero.operator int*(), nullptr);
	EXPECT_EQ(zero.valid(), zero.operator int*() != nullptr);
	zero.free();
}

TEST(Buffer, MoveAssignmentSelfIsSafe) {
	using namespace bench_box;
	using lp_buffer = buffer<uint32_t, lp_flex_allocator_t<uint32_t>>;

	lp_buffer buf(16, lp_flex_allocator_t<uint32_t>(false));
	ASSERT_TRUE(buf.valid());
	auto* const ptr = buf.data;

	lp_buffer& alias = buf;
	buf = std::move(alias);
	EXPECT_TRUE(buf.valid());
	EXPECT_EQ(buf.data, ptr);
	EXPECT_EQ(buf.size, 16u);
}

TEST(Buffer, NonLP) {
	using namespace bench_box;

	lp_flex_allocator_t<uint32_t> alloc(false);

	buffer<uint32_t, lp_flex_allocator_t<uint32_t>> buf(64u * 1024u, alloc);
	ASSERT_NE(buf, nullptr);
	buf.free();
}

TEST(Buffer, DISABLED_LP) { // fails if it can't allocate a large page, may not be an error depending on user privileges
	using namespace bench_box;

	lp_flex_allocator_t<uint32_t> alloc(true);

	buffer<uint32_t, lp_flex_allocator_t<uint32_t>> buf(64u * 1024u, alloc);
	ASSERT_NE(buf, nullptr);
	buf.free();
}
