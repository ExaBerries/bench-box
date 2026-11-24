#pragma once

namespace bench_box {
	namespace simd {

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 create2i64(int64_t m0, int64_t m1) noexcept {
			return _mm_set_epi64x(static_cast<long long>(m1), static_cast<long long>(m0));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE uint32_t maskbits2i64(const reg2i64& mask) noexcept {
			return static_cast<uint32_t>(_mm_movemask_pd(cast2i64_to_2d(mask)));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 mask_from_bits2i64(uint32_t bits) noexcept {
			return create2i64(
				(bits & 0x1u) != 0u ? -1 : 0,
				(bits & 0x2u) != 0u ? -1 : 0
			);
		}

		template <int64_t m0, int64_t m1>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 create_mask_2i64() noexcept {
			alignas(alignof(reg2i64)) static constexpr int64_t mask_data[2] = {
				m0, m1
			};
			return _mm_load_si128(reinterpret_cast<const reg2i64*>(mask_data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 load2i64(const int64_t& data) noexcept {
			return _mm_load_si128(reinterpret_cast<const reg2i64*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 broadcast2i64(const int64_t& value) noexcept {
			return _mm_set1_epi64x(static_cast<long long>(value));
		}

		BENCH_BOX_ALWAYS_INLINE void store2i64(int64_t& out, const reg2i64& reg) noexcept {
			_mm_store_si128(reinterpret_cast<reg2i64*>(&out), reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 zero_reg2i64() noexcept {
			return _mm_setzero_si128();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int64_t extract2i64_lo(const reg2i64& a) noexcept {
			return _mm_cvtsi128_si64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 add2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_add_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 sub2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_sub_epi64(a, b);
		}

		// avx has no 64-bit lane multiply so we need to use 32-bit partial products
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 mul2i64(const reg2i64& a, const reg2i64& b) noexcept {
			const reg2i64 low = _mm_mul_epu32(a, b);
			const reg2i64 a_hi = _mm_srli_epi64(a, 32);
			const reg2i64 b_hi = _mm_srli_epi64(b, 32);
			const reg2i64 cross = _mm_add_epi64(_mm_mul_epu32(a_hi, b), _mm_mul_epu32(a, b_hi));
			return _mm_add_epi64(low, _mm_slli_epi64(cross, 32));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 mulu2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return mul2i64(a, b); // the low 64 bits are identical for signed and unsigned products
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 and2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_and_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 andnot2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_andnot_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 or2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_or_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 xor2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_xor_si128(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 sl2i64(const reg2i64& a) noexcept {
			return _mm_slli_epi64(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 sr2i64(const reg2i64& a) noexcept {
			return _mm_srli_epi64(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cmp_eq2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_cmpeq_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cmp_gt2i64(const reg2i64& a, const reg2i64& b) noexcept {
			return _mm_cmpgt_epi64(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2i64 cvt4i32_to_2i64(const reg4i32& a) noexcept {
			return _mm_cvtepi32_epi64(a);
		}
	} // namespace simd
} // namespace bench_box
