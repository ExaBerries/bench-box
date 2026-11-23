#pragma once
#include <bench_box/math/math.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <new>
#include <iostream>
#include <concepts>

namespace bench_box {
	template <std::integral T>
	[[nodiscard]] constexpr T align_up(T size, T align) noexcept {
		return (size + align - static_cast<T>(1u)) & ~(align - static_cast<T>(1u));
	}

	[[nodiscard]] uint64_t get_available_mem() noexcept;
	[[nodiscard]] std::size_t get_large_page_size() noexcept;

	[[nodiscard]] void* try_malloc_large_page(std::size_t bytes) noexcept;
	void free_large_page(void* ptr, std::size_t bytes) noexcept;

	[[nodiscard]] void* try_malloc_aligned_64(std::size_t bytes) noexcept;
	void free_aligned_64(void* ptr, [[maybe_unused]] std::size_t bytes) noexcept;

	template <typename T>
	struct lp_flex_allocator_t {
		using value_type = T;
		using pointer = T*;
		using const_pointer = const T*;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;

		using propagate_on_container_copy_assignment = std::true_type;
		using propagate_on_container_move_assignment = std::true_type;
		using propagate_on_container_swap = std::true_type;
		using is_always_equal = std::false_type;

		bool use_large_pages = false; // can only be changed before allocating or after deallocating

		lp_flex_allocator_t() noexcept = default;
		explicit lp_flex_allocator_t(bool use_lp) noexcept : use_large_pages(use_lp) {}

		template <typename U>
		lp_flex_allocator_t(const lp_flex_allocator_t<U>& other) noexcept : use_large_pages(other.use_large_pages) {}

		template <typename U>
		struct rebind {
			using other = lp_flex_allocator_t<U>;
		};

		value_type* allocate(size_type n) {
			if (n == 0) {
				return nullptr;
			}
			auto bytes = math::checked_mul(n, sizeof(T));
			if (!bytes) {
				throw std::bad_alloc();
			}
			void* ptr = use_large_pages ? try_malloc_large_page(*bytes) : try_malloc_aligned_64(*bytes);
			if (!ptr) {
				throw std::bad_alloc();
			}
			return static_cast<value_type*>(ptr);
		}

		void deallocate(value_type* ptr, size_type n) noexcept {
			if (!ptr) {
				return;
			}
			auto bytes = math::checked_mul(n, sizeof(T));
			if (!bytes) {
				return;
			}
			if (use_large_pages) {
				free_large_page(ptr, *bytes);
			} else {
				free_aligned_64(ptr, *bytes);
			}
		}

		bool operator==(const lp_flex_allocator_t& other) const noexcept {
			return use_large_pages == other.use_large_pages;
		}

		bool operator!=(const lp_flex_allocator_t& other) const noexcept {
			return !(*this == other);
		}
	};

	template <typename T>
	using lp_flx_allc_t = lp_flex_allocator_t<T>;

	struct unsafe_bump_allocator {
		const bool use_large_pages = false;
		const std::size_t heap_size = 0u;
		void* heap = nullptr;
		void* current = nullptr;

		unsafe_bump_allocator(bool use_lp, std::size_t size_bytes) noexcept : use_large_pages(use_lp), heap_size(size_bytes) {
			heap = use_large_pages ? try_malloc_large_page(size_bytes) : try_malloc_aligned_64(size_bytes);
			current = heap;
		}

		unsafe_bump_allocator(const unsafe_bump_allocator&) noexcept = delete;
		unsafe_bump_allocator(unsafe_bump_allocator&& other) noexcept = delete;

		~unsafe_bump_allocator() noexcept {
			free_heap();
		}

		[[nodiscard]] void* allocate(std::size_t bytes, std::size_t align) noexcept {
			auto curr = reinterpret_cast<uintptr_t>(current);
			auto aligned = align_up(curr, align);
			auto next = aligned + bytes;

			current = reinterpret_cast<void*>(next);

			return reinterpret_cast<void*>(aligned);
		}

		void reset() noexcept {
			current = heap;
		}

		void free_heap() noexcept {
			if (heap != nullptr) {
				if (use_large_pages) {
					free_large_page(heap, heap_size);
				} else {
					free_aligned_64(heap, heap_size);
				}
				current = nullptr;
				heap = nullptr;
			}
		}

		[[nodiscard]] bool heap_valid() const noexcept {
			return heap != nullptr;
		}

		unsafe_bump_allocator& operator=(const unsafe_bump_allocator&) noexcept = delete;
		unsafe_bump_allocator& operator=(unsafe_bump_allocator&& other) noexcept = delete;
	};

	using uns_bump_allc = unsafe_bump_allocator;

	template <typename T>
	struct unsafe_bump_alloc_t {
		using value_type = T;
		using pointer = T*;
		using const_pointer = const T*;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;

		using propagate_on_container_copy_assignment = std::true_type;
		using propagate_on_container_move_assignment = std::true_type;
		using propagate_on_container_swap = std::true_type;
		using is_always_equal = std::false_type;

		unsafe_bump_allocator* alloc = nullptr; // should only ever be changed if its nullptr

		unsafe_bump_alloc_t() noexcept = default;
		explicit unsafe_bump_alloc_t(unsafe_bump_allocator& b_alloc) noexcept : alloc(&b_alloc) {}

		template <typename U>
		unsafe_bump_alloc_t(const unsafe_bump_alloc_t<U>& other) noexcept : alloc(other.alloc) {}

		template <typename U>
		struct rebind {
			using other = unsafe_bump_alloc_t<U>;
		};

		value_type* allocate(size_type n) {
			if (alloc == nullptr || !alloc->heap_valid()) {
				throw std::bad_alloc();
			}
			return reinterpret_cast<T*>(alloc->allocate(sizeof(T) * n, alignof(T)));
		}

		void deallocate([[maybe_unused]] value_type* ptr, [[maybe_unused]] size_type n) noexcept {
			return;
		}

		bool operator==(const unsafe_bump_alloc_t& other) const noexcept {
			return alloc == other.alloc;
		}

		bool operator!=(const unsafe_bump_alloc_t& other) const noexcept {
			return !(*this == other);
		}
	};

	template <typename T>
	using uns_bump_allc_t = unsafe_bump_alloc_t<T>;
} // namespace bench_box
