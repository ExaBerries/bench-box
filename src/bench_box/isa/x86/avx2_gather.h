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

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f gather8f8i32(const float* base, const reg8i32& index) noexcept {
			return _mm256_i32gather_ps(base, index, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d gather4d4i32(const double* base, const reg4i32& index) noexcept {
			return _mm256_i32gather_pd(base, index, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 gather8i328i32(const int32_t* base, const reg8i32& index) noexcept {
			return _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), index, static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 gather4i644i32(const int64_t* base, const reg4i32& index) noexcept {
			return _mm256_i32gather_epi64(reinterpret_cast<const long long*>(base), index, static_cast<int>(sizeof(int64_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f mask_gather8f8i32(const reg8f& src, const float* base, const reg8i32& index, const reg8f& mask) noexcept {
			return _mm256_mask_i32gather_ps(src, base, index, mask, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8f maskz_gather8f8i32(const float* base, const reg8i32& index, const reg8f& mask) noexcept {
			return _mm256_mask_i32gather_ps(zero_reg8f(), base, index, mask, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d mask_gather4d4i32(const reg4d& src, const double* base, const reg4i32& index, const reg4d& mask) noexcept {
			return _mm256_mask_i32gather_pd(src, base, index, mask, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4d maskz_gather4d4i32(const double* base, const reg4i32& index, const reg4d& mask) noexcept {
			return _mm256_mask_i32gather_pd(zero_reg4d(), base, index, mask, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 mask_gather8i328i32(const reg8i32& src, const int32_t* base, const reg8i32& index, const reg8i32& mask) noexcept {
			return _mm256_mask_i32gather_epi32(src, reinterpret_cast<const int*>(base), index, mask, static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 maskz_gather8i328i32(const int32_t* base, const reg8i32& index, const reg8i32& mask) noexcept {
			return _mm256_mask_i32gather_epi32(zero_reg8i32(), reinterpret_cast<const int*>(base), index, mask, static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mask_gather4i644i32(const reg4i64& src, const int64_t* base, const reg4i32& index, const reg4i64& mask) noexcept {
			return _mm256_mask_i32gather_epi64(src, reinterpret_cast<const long long*>(base), index, mask, static_cast<int>(sizeof(int64_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 maskz_gather4i644i32(const int64_t* base, const reg4i32& index, const reg4i64& mask) noexcept {
			return _mm256_mask_i32gather_epi64(zero_reg4i64(), reinterpret_cast<const long long*>(base), index, mask, static_cast<int>(sizeof(int64_t)));
		}
	} // namespace simd
} // namespace bench_box
