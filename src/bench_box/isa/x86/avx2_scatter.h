#pragma once

namespace bench_box {
	namespace simd {
		// Scatter order for duplicate indices is unspecified. Callers must
		// deduplicate if deterministic writes matter.
		BENCH_BOX_ALWAYS_INLINE void scatter8f8i32(float* base, const reg8f& mask, const reg8i32& index, const reg8f& values) noexcept {
			alignas(alignof(reg8i32)) int32_t indices[8] = {};
			alignas(alignof(reg8f)) float vals[8] = {};
			store8i32(indices[0], index);
			store8f(vals[0], values);
			const uint32_t bits = maskbits8f(mask);
			for (int i = 0; i < 8; i++) {
				if ((bits & (1u << static_cast<uint32_t>(i))) != 0u) {
					base[indices[i]] = vals[i];
				}
			}
		}

		BENCH_BOX_ALWAYS_INLINE void scatter4d4i32(double* base, const reg4d& mask, const reg4i32& index, const reg4d& values) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			alignas(alignof(reg4d)) double vals[4] = {};
			store4i32(indices[0], index);
			store4d(vals[0], values);
			const uint32_t bits = maskbits4d(mask);
			for (int i = 0; i < 4; i++) {
				if ((bits & (1u << static_cast<uint32_t>(i))) != 0u) {
					base[indices[i]] = vals[i];
				}
			}
		}

		BENCH_BOX_ALWAYS_INLINE void scatter8i328i32(int32_t* base, const reg8i32& mask, const reg8i32& index, const reg8i32& values) noexcept {
			alignas(alignof(reg8i32)) int32_t indices[8] = {};
			alignas(alignof(reg8i32)) int32_t vals[8] = {};
			store8i32(indices[0], index);
			store8i32(vals[0], values);
			const uint32_t bits = maskbits8i32(mask);
			for (int i = 0; i < 8; i++) {
				if ((bits & (1u << static_cast<uint32_t>(i))) != 0u) {
					base[indices[i]] = vals[i];
				}
			}
		}

		BENCH_BOX_ALWAYS_INLINE void scatter4i644i32(int64_t* base, const reg4i64& mask, const reg4i32& index, const reg4i64& values) noexcept {
			alignas(alignof(reg4i32)) int32_t indices[4] = {};
			alignas(alignof(reg4i64)) int64_t vals[4] = {};
			store4i32(indices[0], index);
			store4i64(vals[0], values);
			const uint32_t bits = maskbits4i64(mask);
			for (int i = 0; i < 4; i++) {
				if ((bits & (1u << static_cast<uint32_t>(i))) != 0u) {
					base[indices[i]] = vals[i];
				}
			}
		}
	} // namespace simd
} // namespace bench_box
