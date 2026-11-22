#pragma once

namespace bench_box {
	namespace simd {
		template <int32_t m0, int32_t m1, int32_t m2, int32_t m3, int32_t m4, int32_t m5, int32_t m6, int32_t m7>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 create_mask_8i32() noexcept {
			alignas(alignof(reg8i32)) static constexpr int32_t mask_data[8] = {
				m0, m1, m2, m3, m4, m5, m6, m7
			};
			return _mm256_load_si256(reinterpret_cast<const reg8i32*>(mask_data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 load8i32(const int32_t& data) noexcept {
			return _mm256_load_si256(reinterpret_cast<const reg8i32*>(&data));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 broadcast8i32(const int32_t& value) noexcept {
			return _mm256_set1_epi32(static_cast<int>(value));
		}

		BENCH_BOX_ALWAYS_INLINE void store8i32(int32_t& out, const reg8i32& reg) noexcept {
			_mm256_store_si256(reinterpret_cast<reg8i32*>(&out), reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 zero_reg8i32() noexcept {
			return _mm256_setzero_si256();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE int32_t extract8i32_lo(const reg8i32& a) noexcept {
			return _mm256_extract_epi32(a, 0);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 add8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_add_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 sub8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_sub_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 mul8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_mullo_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 min8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_min_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 max8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_max_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 minu8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_min_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 maxu8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_max_epu32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 and8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_and_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 andnot8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_andnot_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 or8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_or_si256(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 xor8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_xor_si256(a, b);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 sl8i32(const reg8i32& a) noexcept {
			return _mm256_slli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 sr8i32(const reg8i32& a) noexcept {
			return _mm256_srli_epi32(a, count);
		}

		template <int32_t count>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 sra8i32(const reg8i32& a) noexcept {
			return _mm256_srai_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 slv8i32(const reg8i32& a, const reg8i32& count) noexcept {
			return _mm256_sllv_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 srfv8i32(const reg8i32& a, const reg8i32& count) noexcept {
			return _mm256_srlv_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 srafv8i32(const reg8i32& a, const reg8i32& count) noexcept {
			return _mm256_srav_epi32(a, count);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cmp_lt8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_cmpgt_epi32(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cmp_gt8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_cmpgt_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cmp_eq8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_cmpeq_epi32(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 blendv8i32(const reg8i32& a, const reg8i32& b, const reg8i32& mask) noexcept {
			return _mm256_blendv_epi8(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg4i32 blendv4i32(const reg4i32& a, const reg4i32& b, const reg4i32& mask) noexcept {
			return _mm_blendv_epi8(a, b, mask);
		}

		template <uint32_t mask>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 blend8i32(const reg8i32& a, const reg8i32& b) noexcept {
			return _mm256_blend_epi32(a, b, mask);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 permutevar8i32(const reg8i32& a, const reg8i32& idx) noexcept {
			return _mm256_permutevar8x32_epi32(a, idx);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cvt8f_to_8i32(const reg8f& a) noexcept {
			return _mm256_cvtps_epi32(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg8i32 cvtt8f_to_8i32(const reg8f& a) noexcept {
			return _mm256_cvttps_epi32(a);
		}
	} // namespace simd
} // namespace bench_box
