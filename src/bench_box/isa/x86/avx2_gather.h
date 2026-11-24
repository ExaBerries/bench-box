#pragma once

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f gather4f4i64(const float* base, const reg4i64& index) noexcept {
			return _mm256_i64gather_ps(base, index, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d gather4d4i64(const double* base, const reg4i64& index) noexcept {
			return _mm256_i64gather_pd(base, index, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 gather4i324i64(const int32_t* base, const reg4i64& index) noexcept {
			return _mm256_i64gather_epi32(reinterpret_cast<const int*>(base), index, static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 gather4i644i64(const int64_t* base, const reg4i64& index) noexcept {
			return _mm256_i64gather_epi64(reinterpret_cast<const long long*>(base), index, static_cast<int>(sizeof(int64_t)));
		}
	} // namespace simd
} // namespace bench_box
