#pragma once

namespace bench_box {
	template <typename TYPE, typename ALLOCATOR>
	unique_array<TYPE, ALLOCATOR>::unique_array(std::size_t count, const ALLOCATOR& allocator) noexcept : alloc_(allocator) {
		allocate(count);
	}

	template <typename TYPE, typename ALLOCATOR>
	unique_array<TYPE, ALLOCATOR>::unique_array(unique_array<TYPE, ALLOCATOR>&& source) noexcept(std::is_nothrow_move_constructible_v<ALLOCATOR>) : alloc_(std::move(source.alloc_)), size_(source.size_), data_(source.data_) {
		source.size_ = 0ull;
		source.data_ = nullptr;
	}

	template <typename TYPE, typename ALLOCATOR>
	unique_array<TYPE, ALLOCATOR>& unique_array<TYPE, ALLOCATOR>::operator=(unique_array<TYPE, ALLOCATOR>&& source) noexcept(std::is_nothrow_move_assignable_v<ALLOCATOR>) {
		if (this != &source) {
			free();
			alloc_ = std::move(source.alloc_);
			size_ = source.size_;
			data_ = source.data_;
			source.size_ = 0ull;
			source.data_ = nullptr;
		}
		return *this;
	}

	template <typename TYPE, typename ALLOCATOR>
	unique_array<TYPE, ALLOCATOR>::~unique_array() noexcept {
		free();
	}

	template <typename TYPE, typename ALLOCATOR>
	bool unique_array<TYPE, ALLOCATOR>::valid() const noexcept {
		return data_ != nullptr;
	}

	template <typename TYPE, typename ALLOCATOR>
	bool unique_array<TYPE, ALLOCATOR>::empty() const noexcept {
		return size_ == 0;
	}

	template <typename TYPE, typename ALLOCATOR>
	unique_array<TYPE, ALLOCATOR>::operator bool() const noexcept {
		return !empty();
	}

	template <typename TYPE, typename ALLOCATOR>
	std::size_t unique_array<TYPE, ALLOCATOR>::size() const noexcept {
		return size_;
	}

	template <typename TYPE, typename ALLOCATOR>
	TYPE* unique_array<TYPE, ALLOCATOR>::data() noexcept {
		return data_;
	}

	template <typename TYPE, typename ALLOCATOR>
	const TYPE* unique_array<TYPE, ALLOCATOR>::data() const noexcept {
		return data_;
	}

	template <typename TYPE, typename ALLOCATOR>
	const ALLOCATOR& unique_array<TYPE, ALLOCATOR>::allocator() const noexcept {
		return alloc_;
	}

	template <typename TYPE, typename ALLOCATOR>
	TYPE* unique_array<TYPE, ALLOCATOR>::begin() noexcept {
		return data_;
	}

	template <typename TYPE, typename ALLOCATOR>
	const TYPE* unique_array<TYPE, ALLOCATOR>::begin() const noexcept {
		return data_;
	}

	template <typename TYPE, typename ALLOCATOR>
	TYPE* unique_array<TYPE, ALLOCATOR>::end() noexcept {
		return data_ + size_;
	}

	template <typename TYPE, typename ALLOCATOR>
	const TYPE* unique_array<TYPE, ALLOCATOR>::end() const noexcept {
		return data_ + size_;
	}

	template <typename TYPE, typename ALLOCATOR>
	TYPE& unique_array<TYPE, ALLOCATOR>::operator[](std::size_t idx) noexcept {
		return data_[idx];
	}

	template <typename TYPE, typename ALLOCATOR>
	const TYPE& unique_array<TYPE, ALLOCATOR>::operator[](std::size_t idx) const noexcept {
		return data_[idx];
	}

	template <typename TYPE, typename ALLOCATOR>
	std::span<TYPE> unique_array<TYPE, ALLOCATOR>::span() noexcept {
		return {data_, size_};
	}

	template <typename TYPE, typename ALLOCATOR>
	std::span<const TYPE> unique_array<TYPE, ALLOCATOR>::span() const noexcept {
		return {data_, size_};
	}

	template <typename TYPE, typename ALLOCATOR>
	void unique_array<TYPE, ALLOCATOR>::free() noexcept {
		static_assert(std::is_destructible_v<TYPE>);
		if (data_ == nullptr) {
			return;
		}
		std::destroy_n(data_, size_);
		alloc_.deallocate(data_, size_);
		data_ = nullptr;
		size_ = 0ull;
	}

	template <typename TYPE, typename ALLOCATOR>
	void unique_array<TYPE, ALLOCATOR>::allocate(std::size_t size_n) noexcept {
		static_assert(std::is_default_constructible_v<TYPE>);
		TYPE* ptr = nullptr;
		try {
			ptr = alloc_.allocate(size_n);
			if (size_n != 0) {
				std::uninitialized_default_construct_n(ptr, size_n);
			}
		} catch (...) {
			if (ptr != nullptr) {
				alloc_.deallocate(ptr, size_n);
			}
			data_ = nullptr;
			size_ = 0ull;
			return;
		}
		data_ = ptr;
		size_ = size_n;
	}
} // namespace bench_box
