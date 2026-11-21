#pragma once
#include <bench_box/math/vec2.h>
#include <bench_box/math/vec3.h>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] constexpr TO vector_cast(const vec2<T, A>& vector) noexcept;
		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] constexpr TO vector_cast(const vec3<T, A>& vector) noexcept;

		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] TO round_vector_cast(const vec2<T, A>& vector) noexcept;
		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] TO round_vector_cast(const vec3<T, A>& vector) noexcept;

		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] std::optional<TO> checked_vector_cast(const vec2<T, A>& vector) noexcept;
		template <typename TO, typename T, std::size_t A>
		[[nodiscard]] std::optional<TO> checked_vector_cast(const vec3<T, A>& vector) noexcept;
	} // namespace math
} // namespace bench_box

#include "vec_convertImpl.h"
