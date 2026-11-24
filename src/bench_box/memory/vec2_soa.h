#pragma once
#include <bench_box/math/vec2.h>
#include <bench_box/memory/unique_array.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>

namespace bench_box {
	// Structure-of-arrays storage for vec2 values. The x and y components live in
	// separate contiguous blocks so that per-component work can be vectorized.
	template <typename T, typename ALLOCATOR = std::allocator<T>>
	class vec2_soa {
	public:
		static_assert(std::is_arithmetic<T>::value, "vec2_soa requires an arithmetic element type");
		static_assert(std::is_same<typename ALLOCATOR::value_type, T>::value, "allocator value type must match the element type");

		using element_t = T;
		using allocator_type = ALLOCATOR;
		using vector_t = math::vec2<T>;
		using size_type = std::size_t;
		using storage_type = bench_box::unique_array<T, ALLOCATOR>;

		T* x = nullptr;
		T* y = nullptr;

		// Mutable proxy so soa[index] can be assigned from a vec2, read as a vec2,
		// and have its components modified in place.
		class reference {
		public:
			T& x;
			T& y;

			reference(vec2_soa& owner, size_type index) noexcept;

			reference(const reference&) noexcept = default;

			reference& operator=(const reference& other) noexcept;
			reference& operator=(const vector_t& vector) noexcept;
			reference& operator=(vector_t&& vector) noexcept;

			[[nodiscard]] operator vector_t() const noexcept;

			reference& operator+=(const vector_t& vector) noexcept;
			reference& operator-=(const vector_t& vector) noexcept;
			reference& operator+=(T value) noexcept;
			reference& operator-=(T value) noexcept;
			reference& operator*=(T value) noexcept;
			reference& operator/=(T value) noexcept;
		};

		vec2_soa() noexcept = default;
		explicit vec2_soa(size_type count, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		explicit vec2_soa(std::span<const vector_t> vectors, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		vec2_soa(std::initializer_list<vector_t> vectors, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;

		vec2_soa(const vec2_soa&) = delete;
		vec2_soa& operator=(const vec2_soa&) = delete;

		vec2_soa(vec2_soa&& source) noexcept(std::is_nothrow_move_constructible_v<storage_type>);
		vec2_soa& operator=(vec2_soa&& source) noexcept(std::is_nothrow_move_assignable_v<storage_type>);

		~vec2_soa() = default;

		[[nodiscard]] bool valid() const noexcept;
		[[nodiscard]] bool empty() const noexcept;
		[[nodiscard]] explicit operator bool() const noexcept;
		[[nodiscard]] size_type size() const noexcept;

		[[nodiscard]] const ALLOCATOR& allocator() const noexcept;

		[[nodiscard]] std::span<T> x_span() noexcept;
		[[nodiscard]] std::span<const T> x_span() const noexcept;
		[[nodiscard]] std::span<T> y_span() noexcept;
		[[nodiscard]] std::span<const T> y_span() const noexcept;

		[[nodiscard]] vector_t get(size_type index) const noexcept;
		vec2_soa& set(size_type index, const vector_t& vector) noexcept;
		vec2_soa& set(size_type index, T x, T y) noexcept;

		[[nodiscard]] vector_t operator[](size_type index) const noexcept;
		[[nodiscard]] reference operator[](size_type index) noexcept;

		vec2_soa& fill(const vector_t& vector) noexcept;
		vec2_soa& fill(T x, T y) noexcept;

	private:
		static constexpr size_type components = 2;

		[[nodiscard]] static size_type storage_size(size_type count) noexcept;

		storage_type data_ = storage_type();
		size_type size_ = 0;
	};

	using vec2f_soa = vec2_soa<float>;
	using vec2d_soa = vec2_soa<double>;
	using vec2i_soa = vec2_soa<std::int32_t>;
	using vec2li_soa = vec2_soa<std::int64_t>;
} // namespace bench_box

#include "vec2_soaImpl.h"
