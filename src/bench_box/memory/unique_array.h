#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace bench_box {
	// non-resizeable array with no deep copy and RAII semantics
	template <typename TYPE, typename ALLOCATOR = std::allocator<TYPE>>
	class unique_array {
	public:
		using ElementType = TYPE;
		using AllocatorType = ALLOCATOR;

		unique_array() noexcept = default;
		explicit unique_array(std::size_t count, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;

		unique_array(const unique_array&) = delete;
		unique_array& operator=(const unique_array&) = delete;

		unique_array(unique_array&& source) noexcept(std::is_nothrow_move_constructible_v<ALLOCATOR>);
		unique_array& operator=(unique_array&& source) noexcept(std::is_nothrow_move_assignable_v<ALLOCATOR>);

		~unique_array() noexcept;

		[[nodiscard]] bool valid() const noexcept;
		[[nodiscard]] bool empty() const noexcept;
		[[nodiscard]] explicit operator bool() const noexcept;

		[[nodiscard]] std::size_t size() const noexcept;

		[[nodiscard]] TYPE* data() noexcept;
		[[nodiscard]] const TYPE* data() const noexcept;

		[[nodiscard]] const ALLOCATOR& allocator() const noexcept;

		[[nodiscard]] TYPE* begin() noexcept;
		[[nodiscard]] const TYPE* begin() const noexcept;
		[[nodiscard]] TYPE* end() noexcept;
		[[nodiscard]] const TYPE* end() const noexcept;

		[[nodiscard]] TYPE& operator[](std::size_t idx) noexcept;
		[[nodiscard]] const TYPE& operator[](std::size_t idx) const noexcept;

		[[nodiscard]] std::span<TYPE> span() noexcept;
		[[nodiscard]] std::span<const TYPE> span() const noexcept;

	private:
		void free() noexcept;
		void allocate(std::size_t size_n) noexcept;

		ALLOCATOR alloc_ = ALLOCATOR();
		std::size_t size_ = 0;
		TYPE* data_ = nullptr;
	};

	template <typename TYPE, typename ALLOCATOR = std::allocator<TYPE>>
	using uarray = unique_array<TYPE, ALLOCATOR>;
} // namespace bench_box

#include "unique_arrayImpl.h"
