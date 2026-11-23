#include <gtest/gtest.h>
#include <bench_box/memory/memory.h>
#include <bench_box/memory/buffer.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

using namespace bench_box;

namespace {
	constexpr std::size_t arena_size = 4u * 1024u * 1024u;

	[[nodiscard]] std::uintptr_t addr(const void* ptr) noexcept {
		return reinterpret_cast<std::uintptr_t>(ptr);
	}

	[[nodiscard]] bool in_arena(const unsafe_bump_allocator& arena, const void* ptr, std::size_t bytes) noexcept {
		const auto start = addr(arena.heap);
		const auto current = addr(ptr);
		if (current < start) {
			return false;
		}
		const auto offset = current - start;
		return offset <= arena.heap_size && bytes <= arena.heap_size - offset;
	}

	struct alignas(64) aligned_64 {
		std::uint64_t value;
	};

	struct alignas(128) aligned_128 {
		std::uint64_t value;
	};
} // namespace

static_assert(!std::is_copy_constructible_v<unsafe_bump_allocator>);
static_assert(!std::is_copy_assignable_v<unsafe_bump_allocator>);
static_assert(!std::is_move_constructible_v<unsafe_bump_allocator>);
static_assert(!std::is_move_assignable_v<unsafe_bump_allocator>);

static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::value_type, int>);
static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::pointer, int*>);
static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::const_pointer, const int*>);
static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::size_type, std::size_t>);
static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<unsafe_bump_alloc_t<int>::rebind<long>::other, unsafe_bump_alloc_t<long>>);
static_assert(std::is_same_v<std::allocator_traits<unsafe_bump_alloc_t<int>>::rebind_alloc<long>, unsafe_bump_alloc_t<long>>);
static_assert(unsafe_bump_alloc_t<int>::propagate_on_container_copy_assignment::value);
static_assert(unsafe_bump_alloc_t<int>::propagate_on_container_move_assignment::value);
static_assert(unsafe_bump_alloc_t<int>::propagate_on_container_swap::value);
static_assert(!unsafe_bump_alloc_t<int>::is_always_equal::value);
static_assert(std::is_default_constructible_v<unsafe_bump_alloc_t<int>>);
static_assert(std::is_nothrow_copy_constructible_v<unsafe_bump_alloc_t<int>>);
static_assert(std::is_nothrow_move_assignable_v<unsafe_bump_alloc_t<int>>);

TEST(UnsafeBumpAllocator, FreshArenaIsEmpty) {
	unsafe_bump_allocator arena(false, arena_size);

	ASSERT_TRUE(arena.heap_valid());
	ASSERT_NE(arena.heap, nullptr);
	EXPECT_EQ(arena.current, arena.heap);
	EXPECT_EQ(arena.heap_size, arena_size);
	EXPECT_FALSE(arena.use_large_pages);
}

TEST(UnsafeBumpAllocator, BumpsMonotonically) {
	unsafe_bump_allocator arena(false, arena_size);
	auto* base = static_cast<std::byte*>(arena.current);

	void* first = arena.allocate(13u, 1u);
	void* second = arena.allocate(7u, 1u);
	void* third = arena.allocate(5u, 1u);

	EXPECT_EQ(first, static_cast<void*>(base));
	EXPECT_EQ(second, static_cast<void*>(base + 13));
	EXPECT_EQ(third, static_cast<void*>(base + 20));
	EXPECT_EQ(arena.current, static_cast<void*>(base + 25));
}

TEST(UnsafeBumpAllocator, HonoursAlignment) {
	unsafe_bump_allocator arena(false, arena_size);

	for (const std::size_t align : {std::size_t{1u}, std::size_t{2u}, std::size_t{4u}, std::size_t{8u}, std::size_t{16u}, std::size_t{32u}, std::size_t{64u}, std::size_t{128u}, std::size_t{4096u}}) {
		void* ptr = arena.allocate(3u, align);
		EXPECT_EQ(addr(ptr) % static_cast<std::uintptr_t>(align), 0u) << "align: " << align;
		EXPECT_TRUE(in_arena(arena, ptr, 3u));
	}
}

TEST(UnsafeBumpAllocator, AlignsUpAfterOddSizes) {
	unsafe_bump_allocator arena(false, arena_size);

	void* first = arena.allocate(3u, 1u);
	void* second = arena.allocate(5u, 8u);

	EXPECT_EQ(addr(second), align_up(addr(first) + 3u, static_cast<std::uintptr_t>(8u)));
}

TEST(UnsafeBumpAllocator, ZeroBytesDoesNotAdvance) {
	unsafe_bump_allocator arena(false, arena_size);
	void* before = arena.current;

	void* ptr = arena.allocate(0u, 8u);

	EXPECT_EQ(ptr, arena.heap);
	EXPECT_EQ(arena.current, before);
}

TEST(UnsafeBumpAllocator, ReportsFailedHeapAllocation) {
	unsafe_bump_allocator arena(false, std::size_t{1} << 60);

	EXPECT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.heap, nullptr);
	// allocate() is deliberately not called here: an unsafe bump allocator trusts
	// the heap exists, the caller is responsible for checking heap_valid() first.
}

TEST(UnsafeBumpAllocator, ResetRewindsToHeap) {
	unsafe_bump_allocator arena(false, arena_size);
	void* const heap = arena.heap;

	void* first = arena.allocate(128u, 64u);
	ASSERT_GT(addr(arena.current), addr(heap));

	arena.reset();

	EXPECT_EQ(arena.heap, heap);
	EXPECT_EQ(arena.current, heap);
	EXPECT_TRUE(arena.heap_valid());
	EXPECT_EQ(arena.heap_size, arena_size);
	EXPECT_EQ(arena.allocate(128u, 64u), first); // recycling starts at the beginning
}

TEST(UnsafeBumpAllocator, ResetIsIdempotent) {
	unsafe_bump_allocator arena(false, arena_size);
	(void)arena.allocate(64u, 16u);

	arena.reset();
	arena.reset();
	arena.reset();

	EXPECT_EQ(arena.current, arena.heap);
	EXPECT_TRUE(arena.heap_valid());
}

TEST(UnsafeBumpAllocator, ResetOnFreshArenaChangesNothing) {
	unsafe_bump_allocator arena(false, arena_size);
	void* const heap = arena.heap;

	arena.reset();

	EXPECT_EQ(arena.current, heap);
	EXPECT_EQ(arena.heap, heap);
}

TEST(UnsafeBumpAllocator, ResetReusesAndAliasesMemory) {
	unsafe_bump_allocator arena(false, arena_size);

	auto* first = static_cast<int*>(arena.allocate(sizeof(int) * 4u, alignof(int)));
	for (int i = 0; i < 4; ++i) {
		first[i] = i + 1;
	}

	arena.reset();

	auto* second = static_cast<int*>(arena.allocate(sizeof(int) * 4u, alignof(int)));
	EXPECT_EQ(second, first);
	// reset neither frees nor clears: new allocations alias the old contents
	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(second[i], i + 1);
	}
}

TEST(UnsafeBumpAllocator, ResetSupportsArenaReuse) {
	unsafe_bump_allocator arena(false, arena_size);
	std::uintptr_t first = 0u;

	for (int run = 0; run < 4; ++run) {
		void* ptr = arena.allocate(256u, 64u);
		if (run == 0) {
			first = addr(ptr);
		}
		EXPECT_EQ(addr(ptr), first);
		EXPECT_EQ(addr(arena.current), first + 256u);
		arena.reset();
	}

	EXPECT_EQ(arena.current, arena.heap);
}

TEST(UnsafeBumpAllocator, FreeHeapReleasesStorage) {
	unsafe_bump_allocator arena(false, arena_size);
	ASSERT_TRUE(arena.heap_valid());
	(void)arena.allocate(64u, 8u);

	arena.free_heap();

	EXPECT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.heap, nullptr);
	EXPECT_EQ(arena.current, nullptr);
	EXPECT_EQ(arena.heap_size, arena_size); // size stays, only the block is released
}

TEST(UnsafeBumpAllocator, FreeHeapIsIdempotent) {
	unsafe_bump_allocator arena(false, arena_size);
	ASSERT_TRUE(arena.heap_valid());

	arena.free_heap();
	arena.free_heap();
	arena.free_heap();

	EXPECT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.heap, nullptr);
	EXPECT_EQ(arena.current, nullptr);
}

TEST(UnsafeBumpAllocator, FreeHeapOnFailedArenaIsNoOp) {
	unsafe_bump_allocator arena(false, std::size_t{1} << 60);
	ASSERT_FALSE(arena.heap_valid());

	arena.free_heap();

	EXPECT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.heap, nullptr);
	EXPECT_EQ(arena.current, nullptr);
}

TEST(UnsafeBumpAllocator, DestructorAfterFreeHeapDoesNotDoubleFree) {
	// an explicit free followed by the destructor must not release the block twice
	unsafe_bump_allocator arena(false, arena_size);
	(void)arena.allocate(1u, 1u);
	arena.free_heap();
	SUCCEED();
}

TEST(UnsafeBumpAllocator, ResetAfterFreeHeapDoesNotResurrect) {
	unsafe_bump_allocator arena(false, arena_size);
	arena.free_heap();
	ASSERT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.current, nullptr);

	arena.reset();

	EXPECT_FALSE(arena.heap_valid());
	EXPECT_EQ(arena.heap, nullptr);
	EXPECT_EQ(arena.current, nullptr);
}

TEST(UnsafeBumpAllocator, DISABLED_LargePages) { // fails if it can't allocate a large page, may not be an error depending on user privileges
	const std::size_t page_size = get_large_page_size();
	unsafe_bump_allocator arena(true, page_size == 0u ? arena_size : page_size);

	ASSERT_TRUE(arena.heap_valid());
	EXPECT_TRUE(arena.use_large_pages);

	void* ptr = arena.allocate(64u, 64u);
	EXPECT_TRUE(in_arena(arena, ptr, 64u));
}

TEST(UnsafeBumpAllocT, EqualityFollowsArena) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_allocator other_arena(false, arena_size);

	unsafe_bump_alloc_t<int> a(arena);
	unsafe_bump_alloc_t<int> b(arena);
	unsafe_bump_alloc_t<int> c(other_arena);

	EXPECT_EQ(a, b);
	EXPECT_NE(a, c);

	const auto copy = a;
	EXPECT_EQ(a, copy);
	EXPECT_EQ(a, unsafe_bump_alloc_t<long>(arena));
}

TEST(UnsafeBumpAllocT, DefaultHandleIsUnbound) {
	unsafe_bump_alloc_t<int> first;
	unsafe_bump_alloc_t<int> second;

	// default-constructed allocators must compare equal
	EXPECT_EQ(first, second);
	EXPECT_EQ(first, unsafe_bump_alloc_t<long>());

	// an unbound handle has no arena to allocate from
	EXPECT_THROW(first.allocate(1u), std::bad_alloc);
	int* leaked = nullptr;
	EXPECT_THROW(leaked = std::allocator_traits<unsafe_bump_alloc_t<int>>::allocate(first, 1u), std::bad_alloc);
	EXPECT_EQ(leaked, nullptr);

	// deallocate is a no-op even for an unbound handle
	EXPECT_NO_THROW(first.deallocate(nullptr, 0u));
}

TEST(UnsafeBumpAllocT, FailedArenaHandleThrowsBadAlloc) {
	unsafe_bump_allocator arena(false, std::size_t{1} << 60);
	ASSERT_FALSE(arena.heap_valid());

	unsafe_bump_alloc_t<int> a(arena);

	EXPECT_THROW(a.allocate(1u), std::bad_alloc);
}

TEST(UnsafeBumpAllocT, FreedArenaHandleThrowsBadAlloc) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	ASSERT_NE(a.allocate(4u), nullptr);

	arena.free_heap();
	ASSERT_FALSE(arena.heap_valid());

	// the handle still points at the arena, but the arena no longer owns memory
	EXPECT_THROW(a.allocate(4u), std::bad_alloc);
}

TEST(UnsafeBumpAllocT, BufferWithUnboundAllocatorStaysEmpty) {
	unsafe_bump_alloc_t<int> unbound;

	buffer<int, unsafe_bump_alloc_t<int>> buf(16u, unbound);

	// buffer catches bad_alloc and reports an empty buffer instead of crashing
	EXPECT_EQ(buf.data, nullptr);
	EXPECT_EQ(buf.size, std::size_t{0});
}

TEST(UnsafeBumpAllocT, VectorWithUnboundAllocatorThrowsBadAlloc) {
	std::vector<int, unsafe_bump_alloc_t<int>> vec;

	EXPECT_EQ(vec.size(), std::size_t{0});
	EXPECT_THROW(vec.push_back(1), std::bad_alloc);
	EXPECT_EQ(vec.size(), std::size_t{0});
}

TEST(UnsafeBumpAllocT, AdvancesBySizeTimesCount) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);

	int* first = a.allocate(3u);
	EXPECT_EQ(addr(first), addr(arena.heap));
	EXPECT_EQ(addr(arena.current), addr(arena.heap) + 3u * sizeof(int));

	int* second = a.allocate(1u);
	EXPECT_EQ(addr(second), addr(first) + 3u * sizeof(int));
	EXPECT_EQ(addr(arena.current), addr(arena.heap) + 4u * sizeof(int));
}

TEST(UnsafeBumpAllocT, HonoursTypeAlignment) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<aligned_64> a64(arena);
	unsafe_bump_alloc_t<aligned_128> a128(arena);

	auto* ptr64 = a64.allocate(2u);
	auto* ptr128 = a128.allocate(2u);

	EXPECT_EQ(addr(ptr64) % alignof(aligned_64), 0u);
	EXPECT_EQ(addr(ptr128) % alignof(aligned_128), 0u);
	EXPECT_GE(addr(ptr128), addr(ptr64) + 2u * sizeof(aligned_64));
	EXPECT_TRUE(in_arena(arena, ptr64, 2u * sizeof(aligned_64)));
	EXPECT_TRUE(in_arena(arena, ptr128, 2u * sizeof(aligned_128)));
}

TEST(UnsafeBumpAllocT, DeallocateDoesNotRecycle) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<std::uint64_t> a(arena);

	std::uint64_t* first = a.allocate(8u);
	void* after_first = arena.current;
	a.deallocate(first, 8u);

	std::uint64_t* second = a.allocate(8u);

	// deallocate is a no-op: the bump pointer keeps moving forward
	EXPECT_EQ(second, after_first);
	EXPECT_GT(addr(second), addr(first));
	EXPECT_EQ(addr(arena.current), addr(after_first) + 8u * sizeof(std::uint64_t));
}

TEST(UnsafeBumpAllocT, WorksWithAllocatorTraits) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	using traits = std::allocator_traits<unsafe_bump_alloc_t<int>>;

	int* ptr = traits::allocate(a, 4u);
	ASSERT_NE(ptr, nullptr);
	EXPECT_EQ(addr(ptr), addr(arena.heap));
	EXPECT_TRUE(in_arena(arena, ptr, 4u * sizeof(int)));

	traits::deallocate(a, ptr, 4u);
}

TEST(UnsafeBumpAllocT, BacksBuffer) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);

	buffer<int, unsafe_bump_alloc_t<int>> buf(16u, a);
	ASSERT_NE(buf.data, nullptr);
	ASSERT_EQ(buf.size, std::size_t{16});
	EXPECT_TRUE(in_arena(arena, buf.data, buf.size * sizeof(int)));

	for (std::size_t i = 0; i < buf.size; ++i) {
		buf[i] = static_cast<int>(i) * 3;
	}
	for (std::size_t i = 0; i < buf.size; ++i) {
		EXPECT_EQ(buf[i], static_cast<int>(i) * 3);
	}

	buf.free();
	EXPECT_EQ(buf.data, nullptr);
	EXPECT_EQ(buf.size, std::size_t{0});
	// freeing a buffer does not return memory to the arena
	EXPECT_GT(addr(arena.current), addr(arena.heap));
}

TEST(UnsafeBumpAllocT, BacksVectorThatGrows) {
	unsafe_bump_allocator arena(false, arena_size);
	std::vector<int, unsafe_bump_alloc_t<int>> vec{unsafe_bump_alloc_t<int>(arena)};

	vec.reserve(64u);
	for (int i = 0; i < 256; ++i) {
		vec.push_back(i);
	}

	ASSERT_EQ(vec.size(), std::size_t{256});
	for (int i = 0; i < 256; ++i) {
		EXPECT_EQ(vec[static_cast<std::size_t>(i)], i);
	}
	EXPECT_TRUE(in_arena(arena, vec.data(), vec.size() * sizeof(int)));
	EXPECT_GT(addr(arena.current), addr(arena.heap));
}

TEST(UnsafeBumpAllocT, VectorMoveDoesNotAllocate) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	std::vector<int, unsafe_bump_alloc_t<int>> vec(a);
	vec.assign({1, 2, 3, 4});

	const int* data = vec.data();
	const std::uintptr_t used = addr(arena.current);

	auto moved = std::move(vec);

	EXPECT_EQ(addr(arena.current), used);
	EXPECT_EQ(moved.get_allocator(), a);
	EXPECT_EQ(moved.data(), data);
	ASSERT_EQ(moved.size(), std::size_t{4});
	EXPECT_EQ(moved[3], 4);
	EXPECT_TRUE(in_arena(arena, moved.data(), moved.size() * sizeof(int)));
}

TEST(UnsafeBumpAllocT, VectorMoveAssignStealsSourceArena) {
	unsafe_bump_allocator first_arena(false, arena_size);
	unsafe_bump_allocator second_arena(false, arena_size);
	unsafe_bump_alloc_t<int> first_alloc(first_arena);
	unsafe_bump_alloc_t<int> second_alloc(second_arena);

	std::vector<int, unsafe_bump_alloc_t<int>> source(first_alloc);
	std::vector<int, unsafe_bump_alloc_t<int>> destination(second_alloc);
	source.assign({10, 20, 30});
	destination.assign({1, 2});

	const int* source_data = source.data();
	const std::uintptr_t first_used = addr(first_arena.current);
	const std::uintptr_t second_used = addr(second_arena.current);

	destination = std::move(source);

	// propagate_on_container_move_assignment == true: storage and arena move together
	EXPECT_EQ(destination.get_allocator(), first_alloc);
	EXPECT_EQ(destination.data(), source_data);
	ASSERT_EQ(destination.size(), std::size_t{3});
	EXPECT_EQ(destination[2], 30);
	EXPECT_EQ(addr(first_arena.current), first_used);
	EXPECT_EQ(addr(second_arena.current), second_used);
	EXPECT_TRUE(in_arena(first_arena, destination.data(), destination.size() * sizeof(int)));
}

TEST(UnsafeBumpAllocT, VectorSwapMovesArenaWithStorage) {
	unsafe_bump_allocator first_arena(false, arena_size);
	unsafe_bump_allocator second_arena(false, arena_size);
	unsafe_bump_alloc_t<int> first_alloc(first_arena);
	unsafe_bump_alloc_t<int> second_alloc(second_arena);

	std::vector<int, unsafe_bump_alloc_t<int>> first(first_alloc);
	std::vector<int, unsafe_bump_alloc_t<int>> second(second_alloc);
	first.assign({1, 2, 3});
	second.assign({7, 8});

	const int* first_data = first.data();
	const int* second_data = second.data();

	first.swap(second);

	// the allocator follows the storage, so neither container ends up pointing at
	// memory owned by the other arena
	EXPECT_EQ(first.get_allocator(), second_alloc);
	EXPECT_EQ(second.get_allocator(), first_alloc);
	EXPECT_EQ(first.data(), second_data);
	EXPECT_EQ(second.data(), first_data);
	ASSERT_EQ(first.size(), std::size_t{2});
	ASSERT_EQ(second.size(), std::size_t{3});
	EXPECT_EQ(first[0], 7);
	EXPECT_EQ(second[0], 1);
	EXPECT_TRUE(in_arena(second_arena, first.data(), first.size() * sizeof(int)));
	EXPECT_TRUE(in_arena(first_arena, second.data(), second.size() * sizeof(int)));
}

TEST(UnsafeBumpAllocT, VectorCopyStaysInArena) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	std::vector<int, unsafe_bump_alloc_t<int>> source(a);
	source.assign({4, 5, 6});

	const std::vector<int, unsafe_bump_alloc_t<int>> copy = source;

	EXPECT_EQ(copy.get_allocator(), a);
	EXPECT_NE(copy.data(), source.data());
	ASSERT_EQ(copy.size(), source.size());
	for (std::size_t i = 0; i < copy.size(); ++i) {
		EXPECT_EQ(copy[i], source[i]);
	}
	EXPECT_TRUE(in_arena(arena, copy.data(), copy.size() * sizeof(int)));
	EXPECT_EQ(source[0], 4);
}

TEST(UnsafeBumpAllocT, BufferMoveTransfersStorage) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	buffer<int, unsafe_bump_alloc_t<int>> source(8u, a);
	for (std::size_t i = 0; i < source.size; ++i) {
		source[i] = static_cast<int>(i);
	}

	const int* data = source.data;
	const std::uintptr_t used = addr(arena.current);

	{
		buffer<int, unsafe_bump_alloc_t<int>> destination(std::move(source));

		EXPECT_EQ(destination.data, data);
		EXPECT_EQ(destination.size, std::size_t{8});
		EXPECT_EQ(destination.alloc, a);
		EXPECT_EQ(addr(arena.current), used);
		EXPECT_EQ(source.data, nullptr);
		EXPECT_EQ(source.size, std::size_t{0});
	}

	// the move did not consume arena space, and the arena does not reclaim on destroy
	EXPECT_EQ(addr(arena.current), used);
	EXPECT_TRUE(in_arena(arena, data, 8u * sizeof(int)));
}

TEST(UnsafeBumpAllocT, BufferMoveAssignStealsSourceArena) {
	unsafe_bump_allocator first_arena(false, arena_size);
	unsafe_bump_allocator second_arena(false, arena_size);
	unsafe_bump_alloc_t<int> first_alloc(first_arena);
	unsafe_bump_alloc_t<int> second_alloc(second_arena);

	buffer<int, unsafe_bump_alloc_t<int>> first(8u, first_alloc);
	buffer<int, unsafe_bump_alloc_t<int>> second(8u, second_alloc);
	ASSERT_NE(first.data, nullptr);
	ASSERT_NE(second.data, nullptr);
	int* const source_data = second.data;

	first = std::move(second);

	// the destination released against its old arena, then took the source's
	// arena and storage
	EXPECT_EQ(first.data, source_data);
	EXPECT_EQ(first.alloc, second_alloc);
	EXPECT_EQ(first.size, std::size_t{8});
	EXPECT_EQ(second.data, nullptr);
	EXPECT_EQ(second.size, std::size_t{0});
	EXPECT_TRUE(in_arena(second_arena, source_data, 8u * sizeof(int)));
}

TEST(UnsafeBumpAllocT, ResetReusesArenaForNextIteration) {
	unsafe_bump_allocator arena(false, arena_size);
	unsafe_bump_alloc_t<int> a(arena);
	int* first = nullptr;

	for (int iteration = 0; iteration < 3; ++iteration) {
		buffer<int, unsafe_bump_alloc_t<int>> buf(32u, a);
		ASSERT_NE(buf.data, nullptr);
		if (iteration == 0) {
			first = buf.data;
		}
		EXPECT_EQ(buf.data, first); // same scratch space every iteration
		for (std::size_t i = 0; i < buf.size; ++i) {
			buf[i] = static_cast<int>(i) + iteration;
		}
		EXPECT_EQ(buf[0], iteration);

		arena.reset();
	}

	EXPECT_EQ(arena.current, arena.heap);
	EXPECT_TRUE(arena.heap_valid());
}
