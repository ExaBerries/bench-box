#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

namespace bench_box {
	namespace simd {
		BENCH_BOX_ALWAYS_INLINE void scatter4f4i64(float* base, const mask8& mask, const reg4i64& index, const reg4f& values) noexcept {
			_mm256_mask_i64scatter_ps(base, mask, index, values, static_cast<int>(sizeof(float)));
		}

		BENCH_BOX_ALWAYS_INLINE void scatter4d4i64(double* base, const mask8& mask, const reg4i64& index, const reg4d& values) noexcept {
			_mm256_mask_i64scatter_pd(base, mask, index, values, static_cast<int>(sizeof(double)));
		}

		BENCH_BOX_ALWAYS_INLINE void scatter4i324i64(int32_t* base, const mask8& mask, const reg4i64& index, const reg4i32& values) noexcept {
			_mm256_mask_i64scatter_epi32(base, mask, index, values, static_cast<int>(sizeof(int32_t)));
		}

		BENCH_BOX_ALWAYS_INLINE void scatter4i644i64(int64_t* base, const mask8& mask, const reg4i64& index, const reg4i64& values) noexcept {
			_mm256_mask_i64scatter_epi64(base, mask, index, values, static_cast<int>(sizeof(int64_t)));
		}
	} // namespace simd
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
