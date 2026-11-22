#pragma once

namespace bench_box {
	namespace simd {
		template <int32_t m0, int32_t m1, int32_t m2, int32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 create_mask_4i32() noexcept {
			alignas(alignof(reg4i32)) static constexpr int32_t mask_data[4] = {
				m0, m1, m2, m3
			};
			return _mm_load_si128(reinterpret_cast<const reg4i32*>(mask_data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 load4i32(const int32_t& data) noexcept {
			return _mm_load_si128(reinterpret_cast<const reg4i32*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 broadcast4i32(const int32_t& value) noexcept {
			return _mm_set1_epi32(static_cast<int>(value));
		}

		BENCH_BOX_ALWAYS_INLINE void store4i32(int32_t& out, const reg4i32& reg) noexcept {
			_mm_store_si128(reinterpret_cast<reg4i32*>(&out), reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 zero_reg4i32() noexcept {
			return _mm_setzero_si128();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t extract4i32_lo(const reg4i32& a) noexcept {
			return _mm_cvtsi128_si32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 add4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_add_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 sub4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_sub_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 mul4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_mullo_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 min4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_min_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 max4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_max_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 minu4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_min_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 maxu4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_max_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 and4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_and_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 andnot4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_andnot_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 or4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_or_si128(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 xor4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_xor_si128(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 sl4i32(const reg4i32& a) noexcept {
			return _mm_slli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 sr4i32(const reg4i32& a) noexcept {
			return _mm_srli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 sra4i32(const reg4i32& a) noexcept {
			return _mm_srai_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cmp_lt4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_cmpgt_epi32(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cmp_gt4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_cmpgt_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 cmp_eq4i32(const reg4i32& a, const reg4i32& b) noexcept {
			return _mm_cmpeq_epi32(a, b);
		}

		template <uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 permute4i32(const reg4i32& a) noexcept {
			return _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(a), _MM_SHUFFLE(m3, m2, m1, m0)));
		}
	} // namespace simd
} // namespace bench_box
