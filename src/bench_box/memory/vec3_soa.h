#pragma once
#include <bench_box/math/vec3.h>
#include <bench_box/memory/unique_array.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>

namespace bench_box {
	template <typename T, typename ALLOCATOR = std::allocator<T>>
	class vec3_soa {
	public:
		static_assert(std::is_arithmetic<T>::value, "vec3_soa requires an arithmetic element type");
		static_assert(std::is_same<typename ALLOCATOR::value_type, T>::value, "allocator value type must match the element type");

		using element_t = T;
		using allocator_type = ALLOCATOR;
		using vector_t = math::vec3<T>;
		using size_type = std::size_t;
		using storage_type = bench_box::unique_array<T, ALLOCATOR>;

		T* x = nullptr;
		T* y = nullptr;
		T* z = nullptr;

		class reference {
		public:
			T& x;
			T& y;
			T& z;

			reference(vec3_soa& owner, size_type index) noexcept;
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

		vec3_soa() noexcept = default;
		explicit vec3_soa(size_type count, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		explicit vec3_soa(std::span<const vector_t> vectors, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;
		vec3_soa(std::initializer_list<vector_t> vectors, const ALLOCATOR& allocator = ALLOCATOR()) noexcept;

		vec3_soa(const vec3_soa&) = delete;
		vec3_soa& operator=(const vec3_soa&) = delete;

		vec3_soa(vec3_soa&& source) noexcept(std::is_nothrow_move_constructible_v<storage_type>);
		vec3_soa& operator=(vec3_soa&& source) noexcept(std::is_nothrow_move_assignable_v<storage_type>);

		~vec3_soa() = default;

		[[nodiscard]] bool valid() const noexcept;
		[[nodiscard]] bool empty() const noexcept;
		[[nodiscard]] explicit operator bool() const noexcept;
		[[nodiscard]] size_type size() const noexcept;

		[[nodiscard]] const ALLOCATOR& allocator() const noexcept;

		[[nodiscard]] std::span<T> x_span() noexcept;
		[[nodiscard]] std::span<const T> x_span() const noexcept;
		[[nodiscard]] std::span<T> y_span() noexcept;
		[[nodiscard]] std::span<const T> y_span() const noexcept;
		[[nodiscard]] std::span<T> z_span() noexcept;
		[[nodiscard]] std::span<const T> z_span() const noexcept;

		[[nodiscard]] vector_t get(size_type index) const noexcept;
		vec3_soa& set(size_type index, const vector_t& vector) noexcept;
		vec3_soa& set(size_type index, T nx, T ny, T nz) noexcept;

		[[nodiscard]] vector_t operator[](size_type index) const noexcept;
		[[nodiscard]] reference operator[](size_type index) noexcept;

		vec3_soa& fill(const vector_t& vector) noexcept;
		vec3_soa& fill(T nx, T ny, T nz) noexcept;

	private:
		static constexpr size_type components = 3;

		[[nodiscard]] static size_type storage_size(size_type count) noexcept;

		storage_type data_ = storage_type();
		size_type size_ = 0;
	};

	using vec3f_soa = vec3_soa<float>;
	using vec3d_soa = vec3_soa<double>;
	using vec3i_soa = vec3_soa<std::int32_t>;
	using vec3li_soa = vec3_soa<std::int64_t>;
} // namespace bench_box

#include "vec3_soaImpl.h"
