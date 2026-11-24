#pragma once

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f gather4f4i32(const float* base, const reg4i32& index) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			store4i32(indices[0], index);
			return create4f(base[indices[0]], base[indices[1]], base[indices[2]], base[indices[3]]);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d gather2d4i32(const double* base, const reg4i32& index) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			store4i32(indices[0], index);
			return create2d(base[indices[0]], base[indices[1]]);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 gather4i324i32(const int32_t* base, const reg4i32& index) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			store4i32(indices[0], index);
			return create4i32(base[indices[0]], base[indices[1]], base[indices[2]], base[indices[3]]);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f mask_gather4f4i32(const reg4f& src, const float* base, const reg4i32& index, const reg4f& mask) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			alignas(alignof(reg4f)) float source[4] = {};
			store4i32(indices[0], index);
			store4f(source[0], src);
			const uint32_t bits = maskbits4f(mask);
			return create4f(
				(bits & 0x1u) != 0u ? base[indices[0]] : source[0],
				(bits & 0x2u) != 0u ? base[indices[1]] : source[1],
				(bits & 0x4u) != 0u ? base[indices[2]] : source[2],
				(bits & 0x8u) != 0u ? base[indices[3]] : source[3]
			);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4f maskz_gather4f4i32(const float* base, const reg4i32& index, const reg4f& mask) noexcept {
			return mask_gather4f4i32(zero_reg4f(), base, index, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d mask_gather2d4i32(const reg2d& src, const double* base, const reg4i32& index, const reg2d& mask) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			alignas(alignof(reg2d)) double source[2] = {};
			store4i32(indices[0], index);
			store2d(source[0], src);
			const uint32_t bits = maskbits2d(mask);
			return create2d(
				(bits & 0x1u) != 0u ? base[indices[0]] : source[0],
				(bits & 0x2u) != 0u ? base[indices[1]] : source[1]
			);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d maskz_gather2d4i32(const double* base, const reg4i32& index, const reg2d& mask) noexcept {
			return mask_gather2d4i32(zero_reg2d(), base, index, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 mask_gather4i324i32(const reg4i32& src, const int32_t* base, const reg4i32& index, const reg4i32& mask) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			alignas(alignof(reg4i32)) int32_t source[4] = {};
			store4i32(indices[0], index);
			store4i32(source[0], src);
			const uint32_t bits = maskbits4i32(mask);
			return create4i32(
				(bits & 0x1u) != 0u ? base[indices[0]] : source[0],
				(bits & 0x2u) != 0u ? base[indices[1]] : source[1],
				(bits & 0x4u) != 0u ? base[indices[2]] : source[2],
				(bits & 0x8u) != 0u ? base[indices[3]] : source[3]
			);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 maskz_gather4i324i32(const int32_t* base, const reg4i32& index, const reg4i32& mask) noexcept {
			return mask_gather4i324i32(zero_reg4i32(), base, index, mask);
		}
	} // namespace simd
} // namespace bench_box
