#pragma once

namespace bench_box {
	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::buffer(const ALLOCATOR& allocator) noexcept : alloc(allocator), size(0), data(nullptr) {
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::buffer(std::size_t size_n, const ALLOCATOR& allocator) noexcept : alloc(allocator) {
		static_assert(std::is_default_constructible_v<TYPE>);
		TYPE* ptr = nullptr;
		try {
			ptr = alloc.allocate(size_n);
			if (size_n != 0) {
				std::uninitialized_default_construct_n(ptr, size_n);
			}
		} catch (...) {
			if (ptr != nullptr) {
				alloc.deallocate(ptr, size_n);
			}
			this->data = nullptr;
			this->size = 0ull;
			return;
		}
		this->data = ptr;
		this->size = size_n;
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::buffer(std::nullptr_t, const ALLOCATOR& allocator) noexcept : alloc(allocator), size(0ull), data(nullptr) {
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::buffer(buffer<TYPE, ALLOCATOR>&& source) noexcept : alloc(std::move(source.alloc)), size(source.size), data(source.data) {
		source.size = 0ull;
		source.data = nullptr;
	}

	template <typename TYPE, typename ALLOCATOR>
	void buffer<TYPE, ALLOCATOR>::free() noexcept {
		static_assert(std::is_destructible_v<TYPE>);
		if (data == nullptr) {
			return;
		}
		std::destroy_n(data, size);
		alloc.deallocate(data, size);
		data = nullptr;
		size = 0ull;
	}

	template <typename TYPE, typename ALLOCATOR>
	constexpr TYPE* buffer<TYPE, ALLOCATOR>::begin() noexcept {
		return data;
	}

	template <typename TYPE, typename ALLOCATOR>
	constexpr const TYPE* buffer<TYPE, ALLOCATOR>::begin() const noexcept {
		return data;
	}

	template <typename TYPE, typename ALLOCATOR>
	constexpr TYPE* buffer<TYPE, ALLOCATOR>::end() noexcept {
		return data + size;
	}

	template <typename TYPE, typename ALLOCATOR>
	constexpr const TYPE* buffer<TYPE, ALLOCATOR>::end() const noexcept {
		return data + size;
	}

	template <typename TYPE, typename ALLOCATOR>
	bool buffer<TYPE, ALLOCATOR>::valid() const noexcept {
		return data != nullptr;
	}

	template <typename TYPE, typename ALLOCATOR>
	bool buffer<TYPE, ALLOCATOR>::empty() const noexcept {
		return size == 0;
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::operator bool() const noexcept {
		return !empty();
	}

	template <typename TYPE, typename ALLOCATOR>
	std::span<TYPE> buffer<TYPE, ALLOCATOR>::span() noexcept {
		return {data, size};
	}

	template <typename TYPE, typename ALLOCATOR>
	std::span<const TYPE> buffer<TYPE, ALLOCATOR>::span() const noexcept {
		return {data, size};
	}

	template <typename TYPE, typename ALLOCATOR>
	TYPE& buffer<TYPE, ALLOCATOR>::operator[](std::size_t idx) noexcept {
		return data[idx];
	}

	template <typename TYPE, typename ALLOCATOR>
	const TYPE& buffer<TYPE, ALLOCATOR>::operator[](std::size_t idx) const noexcept {
		return data[idx];
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::operator TYPE*() noexcept {
		return data;
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>::operator const TYPE*() const noexcept {
		return data;
	}

	template <typename TYPE, typename ALLOCATOR>
	template <typename OTHERTYPE>
	buffer<TYPE, ALLOCATOR>::operator OTHERTYPE*() noexcept {
		static_assert(sizeof(OTHERTYPE) % sizeof(TYPE) == 0);
		return static_cast<OTHERTYPE*>(data);
	}

	template <typename TYPE, typename ALLOCATOR>
	template <typename OTHERTYPE>
	buffer<TYPE, ALLOCATOR>::operator const OTHERTYPE*() noexcept {
		static_assert(sizeof(OTHERTYPE) % sizeof(TYPE) == 0);
		return static_cast<const OTHERTYPE*>(data);
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>& buffer<TYPE, ALLOCATOR>::operator=(std::nullptr_t) noexcept {
		size = 0;
		data = nullptr;
		return *this;
	}

	template <typename TYPE, typename ALLOCATOR>
	buffer<TYPE, ALLOCATOR>& buffer<TYPE, ALLOCATOR>::operator=(buffer<TYPE, ALLOCATOR>&& source) noexcept {
		if (this != &source) {
			this->alloc = std::move(source.alloc);
			this->size = source.size;
			this->data = source.data;
			source.size = 0ull;
			source.data = nullptr;
		}
		return *this;
	}

	template <typename TYPE, typename ALLOCATOR>
	bool operator==(const buffer<TYPE, ALLOCATOR>& buffer, const void* ptr) noexcept {
		return buffer.data == ptr;
	}

	template <typename TYPE, typename ALLOCATOR>
	bool operator!=(const buffer<TYPE, ALLOCATOR>& buffer, const void* ptr) noexcept {
		return buffer.data != ptr;
	}
} // namspace bench_box
