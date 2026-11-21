#pragma once
#include <type_traits>

namespace bench_box {
	namespace math {
		template <typename TO, typename T, std::size_t A>
		constexpr TO vector_cast(const vec2<T, A>& vector) noexcept {
			using element_t = typename TO::element_t;
			static_assert(std::is_constructible<TO, element_t, element_t>::value, "the vectors have to hold the same number of elements");
			return {static_cast<element_t>(vector.x), static_cast<element_t>(vector.y)};
		}

		template <typename TO, typename T, std::size_t A>
		constexpr TO vector_cast(const vec3<T, A>& vector) noexcept {
			using element_t = typename TO::element_t;
			static_assert(std::is_constructible<TO, element_t, element_t, element_t>::value, "the vectors have to hold the same number of elements");
			return {static_cast<element_t>(vector.x), static_cast<element_t>(vector.y), static_cast<element_t>(vector.z)};
		}

		template <typename TO, typename T, std::size_t A>
		TO round_vector_cast(const vec2<T, A>& vector) noexcept {
			static_assert(std::is_floating_point<T>::value, "rounding needs a float or double vector");
			using element_t = typename TO::element_t;
			static_assert(std::is_integral<element_t>::value, "rounding goes into an integer vector");
			static_assert(std::is_constructible<TO, element_t, element_t>::value, "the vectors have to hold the same number of elements");
			return {static_cast<element_t>(std::round(vector.x)), static_cast<element_t>(std::round(vector.y))};
		}

		template <typename TO, typename T, std::size_t A>
		TO round_vector_cast(const vec3<T, A>& vector) noexcept {
			static_assert(std::is_floating_point<T>::value, "rounding needs a float or double vector");
			using element_t = typename TO::element_t;
			static_assert(std::is_integral<element_t>::value, "rounding goes into an integer vector");
			static_assert(std::is_constructible<TO, element_t, element_t, element_t>::value, "the vectors have to hold the same number of elements");
			return {static_cast<element_t>(std::round(vector.x)), static_cast<element_t>(std::round(vector.y)), static_cast<element_t>(std::round(vector.z))};
		}

		template <typename TO, typename T, std::size_t A>
		std::optional<TO> checked_vector_cast(const vec2<T, A>& vector) noexcept {
			static_assert(std::is_floating_point<T>::value, "checked_vector_cast needs a float or double vector");
			using element_t = typename TO::element_t;
			static_assert(std::is_integral<element_t>::value && std::is_signed<element_t>::value, "checked_vector_cast goes into an integer vector");
			constexpr double lowest = static_cast<double>(std::numeric_limits<element_t>::lowest());
			constexpr double highest = -lowest;

			const auto fits = [](const T element) noexcept {
				return static_cast<double>(element) >= lowest && static_cast<double>(element) < highest;
			};
			if (!fits(vector.x) || !fits(vector.y)) {
				return std::nullopt;
			}
			return vector_cast<TO>(vector);
		}

		template <typename TO, typename T, std::size_t A>
		std::optional<TO> checked_vector_cast(const vec3<T, A>& vector) noexcept {
			static_assert(std::is_floating_point<T>::value, "checked_vector_cast needs a float or double vector");
			using element_t = typename TO::element_t;
			static_assert(std::is_integral<element_t>::value && std::is_signed<element_t>::value, "checked_vector_cast goes into an integer vector");
			constexpr double lowest = static_cast<double>(std::numeric_limits<element_t>::lowest());
			constexpr double highest = -lowest;

			const auto fits = [](const T element) noexcept {
				return static_cast<double>(element) >= lowest && static_cast<double>(element) < highest;
			};
			if (!fits(vector.x) || !fits(vector.y) || !fits(vector.z)) {
				return std::nullopt;
			}
			return vector_cast<TO>(vector);
		}
	} // namespace math
} // namespace bench_box
