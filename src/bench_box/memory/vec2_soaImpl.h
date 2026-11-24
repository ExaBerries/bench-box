#pragma once
#include <limits>
#include <utility>

namespace bench_box {
	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::reference::reference(vec2_soa& owner, size_type index) noexcept : x(owner.x[index]), y(owner.y[index]) {
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator=(const reference& other) noexcept {
		x = other.x;
		y = other.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator=(const vector_t& vector) noexcept {
		x = vector.x;
		y = vector.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator=(vector_t&& vector) noexcept {
		x = vector.x;
		y = vector.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::reference::operator vector_t() const noexcept {
		return {x, y};
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator+=(const vector_t& vector) noexcept {
		x += vector.x;
		y += vector.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator-=(const vector_t& vector) noexcept {
		x -= vector.x;
		y -= vector.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator+=(T value) noexcept {
		x += value;
		y += value;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator-=(T value) noexcept {
		x -= value;
		y -= value;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator*=(T value) noexcept {
		x *= value;
		y *= value;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference& vec2_soa<T, ALLOCATOR>::reference::operator/=(T value) noexcept {
		x /= value;
		y /= value;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::vec2_soa(size_type count, const ALLOCATOR& allocator) noexcept : data_(storage_size(count), allocator), size_(count) {
		if (count > std::numeric_limits<size_type>::max() / components || data_.size() != count * components) {
			size_ = 0;
		}
		if (size_ != 0) {
			x = data_.data();
			y = x + size_;
		}
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::vec2_soa(std::span<const vector_t> vectors, const ALLOCATOR& allocator) noexcept : vec2_soa(vectors.size(), allocator) {
		T* px = x;
		T* py = y;
		for (size_type i = 0; i < size_; ++i) {
			px[i] = vectors[i].x;
			py[i] = vectors[i].y;
		}
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::vec2_soa(std::initializer_list<vector_t> vectors, const ALLOCATOR& allocator) noexcept
		: vec2_soa(std::span<const vector_t>(vectors.begin(), vectors.size()), allocator) {
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::vec2_soa(vec2_soa&& source) noexcept(std::is_nothrow_move_constructible_v<storage_type>)
		: data_(std::move(source.data_)), size_(source.size_) {
		if (size_ != 0) {
			x = data_.data();
			y = x + size_;
		}
		source.size_ = 0;
		source.x = nullptr;
		source.y = nullptr;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>& vec2_soa<T, ALLOCATOR>::operator=(vec2_soa&& source) noexcept(std::is_nothrow_move_assignable_v<storage_type>) {
		if (this != &source) {
			data_ = std::move(source.data_);
			size_ = source.size_;
			if (size_ != 0) {
				x = data_.data();
				y = x + size_;
			} else {
				x = nullptr;
				y = nullptr;
			}
			source.size_ = 0;
			source.x = nullptr;
			source.y = nullptr;
		}
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	bool vec2_soa<T, ALLOCATOR>::valid() const noexcept {
		return data_.valid();
	}

	template <typename T, typename ALLOCATOR>
	bool vec2_soa<T, ALLOCATOR>::empty() const noexcept {
		return size_ == 0;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>::operator bool() const noexcept {
		return !empty();
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::size_type vec2_soa<T, ALLOCATOR>::size() const noexcept {
		return size_;
	}

	template <typename T, typename ALLOCATOR>
	const ALLOCATOR& vec2_soa<T, ALLOCATOR>::allocator() const noexcept {
		return data_.allocator();
	}

	template <typename T, typename ALLOCATOR>
	std::span<T> vec2_soa<T, ALLOCATOR>::x_span() noexcept {
		return {x, size_};
	}

	template <typename T, typename ALLOCATOR>
	std::span<const T> vec2_soa<T, ALLOCATOR>::x_span() const noexcept {
		return {x, size_};
	}

	template <typename T, typename ALLOCATOR>
	std::span<T> vec2_soa<T, ALLOCATOR>::y_span() noexcept {
		return {y, size_};
	}

	template <typename T, typename ALLOCATOR>
	std::span<const T> vec2_soa<T, ALLOCATOR>::y_span() const noexcept {
		return {y, size_};
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::vector_t vec2_soa<T, ALLOCATOR>::get(size_type index) const noexcept {
		return {x[index], y[index]};
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>& vec2_soa<T, ALLOCATOR>::set(size_type index, const vector_t& vector) noexcept {
		x[index] = vector.x;
		y[index] = vector.y;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>& vec2_soa<T, ALLOCATOR>::set(size_type index, T nx, T ny) noexcept {
		x[index] = nx;
		y[index] = ny;
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::vector_t vec2_soa<T, ALLOCATOR>::operator[](size_type index) const noexcept {
		return get(index);
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::reference vec2_soa<T, ALLOCATOR>::operator[](size_type index) noexcept {
		return reference(*this, index);
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>& vec2_soa<T, ALLOCATOR>::fill(const vector_t& vector) noexcept {
		T* px = x;
		T* py = y;
		for (size_type i = 0; i < size_; ++i) {
			px[i] = vector.x;
			py[i] = vector.y;
		}
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	vec2_soa<T, ALLOCATOR>& vec2_soa<T, ALLOCATOR>::fill(T nx, T ny) noexcept {
		T* px = x;
		T* py = y;
		for (size_type i = 0; i < size_; ++i) {
			px[i] = nx;
			py[i] = ny;
		}
		return *this;
	}

	template <typename T, typename ALLOCATOR>
	typename vec2_soa<T, ALLOCATOR>::size_type vec2_soa<T, ALLOCATOR>::storage_size(size_type count) noexcept {
		if (count > std::numeric_limits<size_type>::max() / components) {
			return 0;
		}
		return count * components;
	}
} // namespace bench_box
