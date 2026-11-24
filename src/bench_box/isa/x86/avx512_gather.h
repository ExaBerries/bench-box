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
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f gather16f16i32(const float* base, const reg16i32& index) noexcept {
			return _mm512_i32gather_ps(index, base, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d gather8d8i32(const double* base, const reg8i32& index) noexcept {
			return _mm512_i32gather_pd(index, base, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 gather16i3216i32(const int32_t* base, const reg16i32& index) noexcept {
			return _mm512_i32gather_epi32(index, reinterpret_cast<const int*>(base), static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 gather8i648i32(const int64_t* base, const reg8i32& index) noexcept {
			return _mm512_i32gather_epi64(index, reinterpret_cast<const long long*>(base), static_cast<int>(sizeof(int64_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f mask_gather16f16i32(const reg16f& src, const float* base, const reg16i32& index, const mask16& mask) noexcept {
			return _mm512_mask_i32gather_ps(src, mask, index, base, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16f maskz_gather16f16i32(const float* base, const reg16i32& index, const mask16& mask) noexcept {
			return _mm512_mask_i32gather_ps(zero_reg16f(), mask, index, base, static_cast<int>(sizeof(float)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d mask_gather8d8i32(const reg8d& src, const double* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm512_mask_i32gather_pd(src, mask, index, base, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8d maskz_gather8d8i32(const double* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm512_mask_i32gather_pd(zero_reg8d(), mask, index, base, static_cast<int>(sizeof(double)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 mask_gather16i3216i32(const reg16i32& src, const int32_t* base, const reg16i32& index, const mask16& mask) noexcept {
			return _mm512_mask_i32gather_epi32(src, mask, index, reinterpret_cast<const int*>(base), static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg16i32 maskz_gather16i3216i32(const int32_t* base, const reg16i32& index, const mask16& mask) noexcept {
			return _mm512_mask_i32gather_epi32(zero_reg16i32(), mask, index, reinterpret_cast<const int*>(base), static_cast<int>(sizeof(int32_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 mask_gather8i648i32(const reg8i64& src, const int64_t* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm512_mask_i32gather_epi64(src, mask, index, reinterpret_cast<const long long*>(base), static_cast<int>(sizeof(int64_t)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i64 maskz_gather8i648i32(const int64_t* base, const reg8i32& index, const mask8& mask) noexcept {
			return _mm512_mask_i32gather_epi64(zero_reg8i64(), mask, index, reinterpret_cast<const long long*>(base), static_cast<int>(sizeof(int64_t)));
		}
	} // namespace simd
} // namespace bench_box

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
