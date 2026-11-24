#pragma once

namespace bench_box {
	namespace simd {
		template <int64_t m0, int64_t m1, int64_t m2, int64_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 create_mask_4i64() noexcept {
			alignas(alignof(reg4i64)) static constexpr int64_t mask_data[4] = {
				m0, m1, m2, m3
			};
			return _mm256_load_si256(reinterpret_cast<const reg4i64*>(mask_data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 load4i64(const int64_t& data) noexcept {
			return _mm256_load_si256(reinterpret_cast<const reg4i64*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 broadcast4i64(const int64_t& value) noexcept {
			return _mm256_set1_epi64x(static_cast<long long>(value));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 create4i64(int64_t m0, int64_t m1, int64_t m2, int64_t m3) noexcept {
			return _mm256_set_epi64x(
				static_cast<long long>(m3),
				static_cast<long long>(m2),
				static_cast<long long>(m1),
				static_cast<long long>(m0)
			);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE uint32_t maskbits4i64(const reg4i64& mask) noexcept {
			return static_cast<uint32_t>(_mm256_movemask_pd(cast4i64_to_4d(mask)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mask_from_bits4i64(uint32_t bits) noexcept {
			return create4i64(
				(bits & 0x1u) != 0u ? -1 : 0,
				(bits & 0x2u) != 0u ? -1 : 0,
				(bits & 0x4u) != 0u ? -1 : 0,
				(bits & 0x8u) != 0u ? -1 : 0
			);
		}

		BENCH_BOX_ALWAYS_INLINE void store4i64(int64_t& out, const reg4i64& reg) noexcept {
			_mm256_store_si256(reinterpret_cast<reg4i64*>(&out), reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 zero_reg4i64() noexcept {
			return _mm256_setzero_si256();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int64_t extract4i64_lo(const reg4i64& a) noexcept {
			return _mm256_extract_epi64(a, 0);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 reg4i642i64_lo(const reg4i64& a) noexcept {
			return _mm256_castsi256_si128(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 reg4i642i64_hi(const reg4i64& a) noexcept {
			return _mm256_extracti128_si256(a, 1);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 reg2i644i64(const reg2i64& a) noexcept {
			return _mm256_inserti128_si256(_mm256_castsi128_si256(a), a, 1);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 add4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_add_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 sub4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_sub_epi64(a, b);
		}

		// avx12has no 64-bit lane multiply so we need to use 32-bit partial products
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mul4i64(const reg4i64& a, const reg4i64& b) noexcept {
			const reg4i64 low = _mm256_mul_epu32(a, b);
			const reg4i64 a_hi = _mm256_srli_epi64(a, 32);
			const reg4i64 b_hi = _mm256_srli_epi64(b, 32);
			const reg4i64 cross = _mm256_add_epi64(_mm256_mul_epu32(a_hi, b), _mm256_mul_epu32(a, b_hi));
			return _mm256_add_epi64(low, _mm256_slli_epi64(cross, 32));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 mulu4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return mul4i64(a, b); // the low 64 bits are identical for signed and unsigned products
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 and4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_and_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 andnot4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_andnot_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 or4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_or_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 xor4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_xor_si256(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 sl4i64(const reg4i64& a) noexcept {
			return _mm256_slli_epi64(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 sr4i64(const reg4i64& a) noexcept {
			return _mm256_srli_epi64(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cmp_eq4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_cmpeq_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cmp_gt4i64(const reg4i64& a, const reg4i64& b) noexcept {
			return _mm256_cmpgt_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 blendv4i64(const reg4i64& a, const reg4i64& b, const reg4i64& mask) noexcept {
			return _mm256_blendv_epi8(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 blendv2i64(const reg2i64& a, const reg2i64& b, const reg2i64& mask) noexcept {
			return _mm_blendv_epi8(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i64 cvt4i32_to_4i64(const reg4i32& a) noexcept {
			return _mm256_cvtepi32_epi64(a);
		}
	} // namespace simd
} // namespace bench_box
