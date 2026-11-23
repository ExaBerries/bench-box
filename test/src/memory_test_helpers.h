#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace bb_test {
	struct allocation_counts {
		std::size_t allocations = 0;
		std::size_t deallocations = 0;
	};

	template <typename T>
	struct counting_allocator {
		using value_type = T;

		allocation_counts* counts = nullptr;

		counting_allocator() noexcept = default;
		explicit counting_allocator(allocation_counts& c) noexcept : counts(&c) {}

		template <typename U>
		counting_allocator(const counting_allocator<U>& other) noexcept : counts(other.counts) {}

		[[nodiscard]] T* allocate(std::size_t n) {
			++counts->allocations;
			return std::allocator<T>{}.allocate(n);
		}

		void deallocate(T* ptr, std::size_t n) noexcept {
			++counts->deallocations;
			std::allocator<T>{}.deallocate(ptr, n);
		}

		bool operator==(const counting_allocator&) const noexcept {
			return true;
		}

		bool operator!=(const counting_allocator&) const noexcept {
			return false;
		}
	};

	// Always owns a block, even for a zero-element request.
	template <typename T>
	struct nonnull_zero_allocator {
		using value_type = T;

		[[nodiscard]] T* allocate(std::size_t n) {
			return std::allocator<T>{}.allocate(n == 0 ? 1 : n);
		}

		void deallocate(T* ptr, std::size_t n) noexcept {
			std::allocator<T>{}.deallocate(ptr, n == 0 ? 1 : n);
		}

		bool operator==(const nonnull_zero_allocator&) const noexcept {
			return true;
		}

		bool operator!=(const nonnull_zero_allocator&) const noexcept {
			return false;
		}
	};

	struct throwing_default_ctor {
		throwing_default_ctor() {
			throw std::bad_alloc{};
		}

		~throwing_default_ctor() noexcept = default;
	};

	struct throwing_move_allocator {
		using value_type = int;

		throwing_move_allocator() noexcept = default;
		throwing_move_allocator(const throwing_move_allocator&) noexcept = default;
		throwing_move_allocator& operator=(const throwing_move_allocator&) noexcept = default;
		throwing_move_allocator(throwing_move_allocator&&) noexcept(false) {}
		throwing_move_allocator& operator=(throwing_move_allocator&&) noexcept(false) {
			return *this;
		}

		[[nodiscard]] int* allocate(std::size_t n) {
			return std::allocator<int>{}.allocate(n);
		}

		void deallocate(int* ptr, std::size_t n) noexcept {
			std::allocator<int>{}.deallocate(ptr, n);
		}

		bool operator==(const throwing_move_allocator&) const noexcept {
			return true;
		}

		bool operator!=(const throwing_move_allocator&) const noexcept {
			return false;
		}
	};
} // namespace bb_test
