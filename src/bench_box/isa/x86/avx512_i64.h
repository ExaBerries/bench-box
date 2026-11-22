#pragma once

namespace bench_box {
	namespace simd {
		template <int64_t m0, int64_t m1, int64_t m2, int64_t m3, int64_t m4, int64_t m5, int64_t m6, int64_t m7>
		[[gnu::always_inline, nodiscard]] inline reg8i64 create_mask_8i64() noexcept {
			alignas(alignof(reg8i64)) static constexpr int64_t mask_data[8] = {
				m0, m1, m2, m3, m4, m5, m6, m7
			};
			return _mm512_load_si512(reinterpret_cast<const reg8i64*>(mask_data));
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 load8i64(const int64_t& data) noexcept {
			return _mm512_load_si512(reinterpret_cast<const reg8i64*>(&data));
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 broadcast8i64(const int64_t& value) noexcept {
			return _mm512_set1_epi64(static_cast<long long>(value));
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 mask_load8i64(const reg8i64& src, const int64_t& data, const mask8& k) noexcept {
			return _mm512_mask_load_epi64(src, k, &data);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 maskz_load8i64(const int64_t& data, const mask8& k) noexcept {
			return _mm512_maskz_load_epi64(k, &data);
		}

		[[gnu::always_inline]] inline void store8i64(int64_t& out, const reg8i64& reg) noexcept {
			_mm512_store_si512(reinterpret_cast<reg8i64*>(&out), reg);
		}

		[[gnu::always_inline]] inline void mask_store8i64(int64_t& out, const mask8& k, const reg8i64& reg) noexcept {
			_mm512_mask_store_epi64(&out, k, reg);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 zero_reg8i64() noexcept {
			return _mm512_setzero_si512();
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 add8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_add_epi64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 sub8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_sub_epi64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 min8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_min_epi64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 max8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_max_epi64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 minu8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_min_epu64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 maxu8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_max_epu64(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 abs8i64(const reg8i64& a) noexcept {
			return _mm512_abs_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 and8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_and_si512(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 andnot8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_andnot_si512(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 or8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_or_si512(a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 xor8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_xor_si512(a, b);
		}

		template <int32_t count>
		[[gnu::always_inline, nodiscard]] inline reg8i64 sl8i64(const reg8i64& a) noexcept {
			return _mm512_slli_epi64(a, count);
		}

		template <int32_t count>
		[[gnu::always_inline, nodiscard]] inline reg8i64 sr8i64(const reg8i64& a) noexcept {
			return _mm512_srli_epi64(a, count);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 srafv8i64(const reg8i64& a, const reg8i64& count) noexcept {
			return _mm512_srav_epi64(a, count);
		}

		template <int32_t count>
		[[gnu::always_inline, nodiscard]] inline reg8i64 sra8i64(const reg8i64& a) noexcept {
			return _mm512_srai_epi64(a, count);
		}

		[[gnu::always_inline, nodiscard]] inline int64_t reduce_min8i64(const reg8i64& a) noexcept {
			return _mm512_reduce_min_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline int64_t reduce_max8i64(const reg8i64& a) noexcept {
			return _mm512_reduce_max_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline uint64_t reduce_minu8i64(const reg8i64& a) noexcept {
			return static_cast<uint64_t>(_mm512_reduce_min_epu64(a));
		}

		[[gnu::always_inline, nodiscard]] inline uint64_t reduce_maxu8i64(const reg8i64& a) noexcept {
			return static_cast<uint64_t>(_mm512_reduce_max_epu64(a));
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 slv8i64(const reg8i64& a, const reg8i64& count) noexcept {
			return _mm512_sllv_epi64(a, count);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 srfv8i64(const reg8i64& a, const reg8i64& count) noexcept {
			return _mm512_srlv_epi64(a, count);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_eq8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_neq8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NE);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_lt8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_LT);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_le8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_LE);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_gt8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLE);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmp_ge8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmpu_lt8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmpu_le8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LE);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmpu_gt8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLE);
		}

		[[gnu::always_inline, nodiscard]] inline mask8 cmpu_ge8i64(const reg8i64& a, const reg8i64& b) noexcept {
			return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 blend8i64mask(const reg8i64& a, const reg8i64& b, const mask8& mask) noexcept {
			return _mm512_mask_blend_epi64(mask, a, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 permutexvar8i64(const reg8i64& a, const reg8i64& idx) noexcept {
			return _mm512_permutexvar_epi64(idx, a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 permutex2var8i64(const reg8i64& a, const reg8i64& b, const reg8i64& idx) noexcept {
			return _mm512_permutex2var_epi64(a, idx, b);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvt8i32_to_8i64(const reg8i32& a) noexcept {
			return _mm512_cvtepi32_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline reg8i64 cvtu8i32_to_8i64(const reg8i32& a) noexcept {
			return _mm512_cvtepu32_epi64(a);
		}

		[[gnu::always_inline, nodiscard]] inline int64_t reduce_add8i64(const reg8i64& a) noexcept {
			return _mm512_reduce_add_epi64(a);
		}
	} // namespace simd
} // namespace bench_box
