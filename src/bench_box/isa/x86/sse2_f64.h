#pragma once

namespace bench_box {
	namespace simd {
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d load2d2d(const double& data) noexcept {
			return _mm_load_pd(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d loadu2d2d(const double& data) noexcept {
			return _mm_loadu_pd(&data);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d load1d2d(const double& value) noexcept {
			return _mm_load1_pd(&value);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d load1d2d_lo(const double& value) noexcept {
			return _mm_load_sd(&value);
		}

		BENCH_BOX_ALWAYS_INLINE void store2d(double& out, const reg2d& reg) noexcept {
			_mm_store_pd(&out, reg);
		}

		BENCH_BOX_ALWAYS_INLINE void storeu2d(double& out, const reg2d& reg) noexcept {
			_mm_storeu_pd(&out, reg);
		}

		BENCH_BOX_ALWAYS_INLINE void store2d_lo(double& out, const reg2d& reg) noexcept {
			_mm_store_sd(&out, reg);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d zero_reg2d() noexcept {
			return _mm_setzero_pd();
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE double extract2d_lo(const reg2d& a) noexcept {
			return _mm_cvtsd_f64(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d add2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_add_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d sub2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_sub_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d mul2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_mul_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d div2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_div_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d sqrt2d(const reg2d& a) noexcept {
			return _mm_sqrt_pd(a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d min2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_min_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d max2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_max_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d abs2d(const reg2d& a) noexcept {
			return _mm_andnot_pd(_mm_set1_pd(-0.0), a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d neg2d(const reg2d& a) noexcept {
			return _mm_xor_pd(a, _mm_set1_pd(-0.0));
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d and2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_and_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d andnot2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_andnot_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d or2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_or_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d xor2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_xor_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_lt2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmplt_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_le2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmple_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_eq2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmpeq_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_gt2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmplt_pd(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_ge2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmple_pd(b, a);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d cmp_neq2d(const reg2d& a, const reg2d& b) noexcept {
			return _mm_cmpneq_pd(a, b);
		}

		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d blendv2d(const reg2d& a, const reg2d& b, const reg2d& mask) noexcept {
			#if defined(__SSE4_1__)
				return _mm_blendv_pd(a, b, mask);
			#else
				return or2d(and2d(mask, b), andnot2d(mask, a));
			#endif
		}

		template <int32_t m0, int32_t m1>
		[[nodiscard]] BENCH_BOX_ALWAYS_INLINE reg2d permute2d(const reg2d& a) noexcept {
			return _mm_shuffle_pd(a, a, ((m1 & 1) << 1) | (m0 & 1));
		}
	} // namespace simd
} // namespace bench_box
