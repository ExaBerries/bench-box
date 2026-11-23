#pragma once
#include "sse2.h"

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f load8f8f(const float& data) noexcept {
			return _mm256_load_ps(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f loadu8f8f(const float& data) noexcept {
			return _mm256_loadu_ps(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f load1f8f(const float& value) noexcept {
			return _mm256_set1_ps(value);
		}

		BENCH_BOX_ALWAYS_INLINE void store8f(float& out, const reg8f& reg) noexcept {
			_mm256_store_ps(&out, reg);
		}

		BENCH_BOX_ALWAYS_INLINE void storeu8f(float& out, const reg8f& reg) noexcept {
			_mm256_storeu_ps(&out, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f zero_reg8f() noexcept {
			return _mm256_setzero_ps();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f add8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_add_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f sub8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_sub_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f mul8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_mul_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f div8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_div_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f sqrt8f(const reg8f& a) noexcept {
			return _mm256_sqrt_ps(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f min8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_min_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f max8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_max_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f abs8f(const reg8f& a) noexcept {
			return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f neg8f(const reg8f& a) noexcept {
			return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f and8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_and_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f andnot8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_andnot_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f or8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_or_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f xor8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_xor_ps(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_lt8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_le8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_eq8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_gt8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(b, a, _CMP_LT_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_ge8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(b, a, _CMP_LE_OS);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f cmp_neq8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f blendv8f(const reg8f& a, const reg8f& b, const reg8f& mask) noexcept {
			return _mm256_blendv_ps(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f floor8f(const reg8f& a) noexcept {
			return _mm256_floor_ps(a);
		}

		template <uint32_t mask>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f blend8f(const reg8f& a, const reg8f& b) noexcept {
			return _mm256_blend_ps(a, b, static_cast<int>(mask));
		}
	} // namespace simd
} // namespace bench_box
