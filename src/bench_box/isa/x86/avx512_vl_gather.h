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
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f mask_gather8f8i32(const reg8f& src, const float* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_ps(src, mask, index, base, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f maskz_gather8f8i32(const float* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_ps(zero_reg8f(), mask, index, base, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d mask_gather4d4i32(const reg4d& src, const double* base, const reg4i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_pd(src, mask, index, base, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d maskz_gather4d4i32(const double* base, const reg4i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_pd(zero_reg4d(), mask, index, base, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 mask_gather8i328i32(const reg8i32& src, const int32_t* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_epi32(src, mask, index, reinterpret_cast<const int*>(base), static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 maskz_gather8i328i32(const int32_t* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_epi32(zero_reg8i32(), mask, index, reinterpret_cast<const int*>(base), static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mask_gather4i644i32(const reg4i64& src, const int64_t* base, const reg4i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_epi64(src, mask, index, reinterpret_cast<const long long*>(base), static_cast<int>(sizeof(int64_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 maskz_gather4i644i32(const int64_t* base, const reg4i32& index, const mask8& mask) noexcept {
			return _mm256_mmask_i32gather_epi64(zero_reg4i64(), mask, index, reinterpret_cast<const long long*>(base), static_cast<int>(sizeof(int64_t)));
		}
	} // namespace simd
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
