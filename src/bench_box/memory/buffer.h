#pragma once
#include <cstddef>
#include <memory>
#include <cstdint>
#include <span>
#include <type_traits>

namespace bench_box {
	// unsafe by design heap array with no deep copy that presents itself as essentially a thin pointer with limited c++ semantics and no RAII
	template <typename TYPE, typename ALLOCATOR = std::allocator<TYPE>>
	struct buffer {
		static_assert(std::is_nothrow_move_constructible_v<ALLOCATOR>);
		static_assert(std::is_nothrow_move_assignable_v<ALLOCATOR>);
		static_assert(std::is_nothrow_destructible_v<TYPE>);

		using ElementType = TYPE;
		using AllocatorType = ALLOCATOR;
		ALLOCATOR alloc = ALLOCATOR(); // only re-assign while empty
		std::size_t size = 0; // must be 0 when data is nullptr
		TYPE* data = nullptr;

		buffer() noexcept = default;
		buffer(const ALLOCATOR& allocator) noexcept;
		buffer(std::size_t size_n, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		buffer(std::nullptr_t, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		buffer(const buffer&) noexcept = delete;
		buffer(buffer&& source) noexcept;
		~buffer() noexcept = default;

		void free() noexcept;

		[[nodiscard]] constexpr TYPE* begin() noexcept;
		[[nodiscard]] constexpr const TYPE* begin() const noexcept;
		[[nodiscard]] constexpr TYPE* end() noexcept;
		[[nodiscard]] constexpr const TYPE* end() const noexcept;

		[[nodiscard]] bool valid() const noexcept;
		[[nodiscard]] bool empty() const noexcept;
		[[nodiscard]] explicit operator bool() const noexcept;

		[[nodiscard]] std::span<TYPE> span() noexcept;
		[[nodiscard]] std::span<const TYPE> span() const noexcept;

		[[nodiscard]] TYPE& operator[](std::size_t idx) noexcept;
		[[nodiscard]] const TYPE& operator[](std::size_t idx) const noexcept;

		[[nodiscard]] operator TYPE*() noexcept;
		[[nodiscard]] operator const TYPE*() const noexcept;
		template <typename OTHERTYPE>
		[[nodiscard]] explicit operator OTHERTYPE*() noexcept;
		template <typename OTHERTYPE>
		[[nodiscard]] explicit operator const OTHERTYPE*() noexcept;

		buffer& operator=(std::nullptr_t) noexcept;
		buffer& operator=(const buffer&) = delete;
		buffer& operator=(buffer&&) noexcept;
	};

	template <typename TYPE, typename ALLOCATOR>
	[[nodiscard]] bool operator==(const buffer<TYPE, ALLOCATOR>& buffer, const void* ptr) noexcept;

	template <typename TYPE, typename ALLOCATOR>
	[[nodiscard]] bool operator!=(const buffer<TYPE, ALLOCATOR>& buffer, const void* ptr) noexcept;
} // namespace bench_box

#include "bufferImpl.h"
