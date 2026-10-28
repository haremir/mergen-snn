# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

cdef extern from "<immintrin.h>":
    # AVX2 and FMA intrinsics
    ctypedef float __m256
    ctypedef float __m128
    __m256 _mm256_setzero_ps() nogil
    __m256 _mm256_loadu_ps(const float* mem_addr) nogil
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) nogil
    __m256 _mm256_add_ps(__m256 a, __m256 b) nogil
    __m128 _mm256_castps256_ps128(__m256 a) nogil
    __m128 _mm256_extractf128_ps(__m256 a, const int imm8) nogil
    __m128 _mm_add_ps(__m128 a, __m128 b) nogil
    __m128 _mm_movehdup_ps(__m128 a) nogil
    __m128 _mm_movehl_ps(__m128 a, __m128 b) nogil
    __m128 _mm_add_ss(__m128 a, __m128 b) nogil
    float _mm_cvtss_f32(__m128 a) nogil

cdef inline float dot_product_avx2(const float* a, const float* b, int k) noexcept nogil:
    cdef __m256 sum0 = _mm256_setzero_ps()
    cdef __m256 sum1 = _mm256_setzero_ps()
    cdef __m256 va0, vb0, va1, vb1
    cdef int i = 0

    while i <= k - 16:
        va0 = _mm256_loadu_ps(a + i)
        vb0 = _mm256_loadu_ps(b + i)
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0)

        va1 = _mm256_loadu_ps(a + i + 8)
        vb1 = _mm256_loadu_ps(b + i + 8)
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1)
        i += 16

    cdef __m256 sum256 = _mm256_add_ps(sum0, sum1)
    while i <= k - 8:
        va0 = _mm256_loadu_ps(a + i)
        vb0 = _mm256_loadu_ps(b + i)
        sum256 = _mm256_fmadd_ps(va0, vb0, sum256)
        i += 8

    cdef __m128 vlow = _mm256_castps256_ps128(sum256)
    cdef __m128 vhigh = _mm256_extractf128_ps(sum256, 1)
    vlow = _mm_add_ps(vlow, vhigh)
    cdef __m128 shuf = _mm_movehdup_ps(vlow)
    cdef __m128 sums = _mm_add_ps(vlow, shuf)
    shuf = _mm_movehl_ps(shuf, sums)
    sums = _mm_add_ss(sums, shuf)
    cdef float total = _mm_cvtss_f32(sums)

    while i < k:
        total += a[i] * b[i]
        i += 1

    return total


def fast_gemm_forward(float[:, :] A, float[:, :] B, object C=None, object out=None, object transposed_b=None):
    """
    AVX2 SIMD ve OpenMP çoklu iş parçacığıyla hızlandırılmış matris çarpımı (GEMM).
    A: (M, K) girdi matrisi
    B: (N, K) ağırlık matrisi (PyTorch Linear) veya (K, N) matrisi (torch.matmul)
    C: (N) bias vektörü (isteğe bağlı)
    out: (M, N) önceden ayrılmış çıktı tamponu (isteğe bağlı)
    transposed_b: True ise B (N, K) PyTorch Linear formatında kabul edilir;
                  False ise B (K, N) standart torch.matmul formatında kabul edilir.
    """
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N
    cdef bint is_transposed = True

    if transposed_b is not None:
        is_transposed = bool(transposed_b)
        if is_transposed:
            N = B.shape[0]
            if B.shape[1] != K:
                raise ValueError(f"Boyut uyumsuzluğu: A ({M}, {K}) ve B ({B.shape[0]}, {B.shape[1]}) çarpılamaz.")
        else:
            N = B.shape[1]
            if B.shape[0] != K:
                raise ValueError(f"Boyut uyumsuzluğu: A ({M}, {K}) ve B ({B.shape[0]}, {B.shape[1]}) çarpılamaz.")
    else:
        # Otomatik algılama
        if B.shape[1] == K and B.shape[0] != K:
            is_transposed = True
            N = B.shape[0]
        elif B.shape[0] == K and B.shape[1] != K:
            is_transposed = False
            N = B.shape[1]
        elif B.shape[0] == K and B.shape[1] == K:
            # Kare matris: bias varsa Linear (transposed), yoksa matmul kabul edilir
            is_transposed = (C is not None)
            N = K
        else:
            raise ValueError(f"Boyut uyumsuzluğu: A ({M}, {K}) ve B ({B.shape[0]}, {B.shape[1]}) çarpılamaz.")

    # Eğer B transposed değilse, AVX2 ardışık bellek erişimi için B transpozesini alıyoruz
    cdef float[:, :] b_view
    if not is_transposed:
        B_t = np.ascontiguousarray(np.asarray(B).T, dtype=np.float32)
        b_view = B_t
    else:
        b_view = B

    if out is None:
        out_arr = np.empty((M, N), dtype=np.float32)
    else:
        out_arr = out
    cdef float[:, :] out_view = out_arr

    cdef const float* a_ptr = &A[0, 0]
    cdef const float* b_ptr = &b_view[0, 0]
    cdef float* out_ptr = &out_view[0, 0]

    # Bias işaretçisi
    cdef const float* bias_ptr = NULL
    cdef bint has_bias = False
    cdef float[:] bias_view
    if C is not None:
        if isinstance(C, np.ndarray) and C.ndim == 1:
            bias_view = C
            bias_ptr = &bias_view[0]
            has_bias = True
        elif hasattr(C, "numpy"):
            c_np = C.detach().contiguous().numpy()
            if c_np.ndim == 1:
                bias_view = c_np
                bias_ptr = &bias_view[0]
                has_bias = True

    cdef int i, j
    cdef const float* a_row
    cdef const float* b_row
    cdef float* out_row

    for i in prange(M, nogil=True):
        a_row = a_ptr + i * K
        out_row = out_ptr + i * N
        for j in range(N):
            b_row = b_ptr + j * K
            if has_bias:
                out_row[j] = dot_product_avx2(a_row, b_row, K) + bias_ptr[j]
            else:
                out_row[j] = dot_product_avx2(a_row, b_row, K)

    return out_arr
