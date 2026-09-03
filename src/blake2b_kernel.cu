/**
 * @file blake2b_kernel.cu
 * @brief Highly optimized CUDA kernel for Blake2b PoW with Constant Memory message broadcast and PTX.
 */

#include "blake2b_cuda.cuh"

// Maximum matches per kernel launch
#define MAX_FOUND_NONCES 16

__constant__ blake2b_midstate_t d_midstate;

/**
 * @brief CUDA kernel for parallel Blake2b nonce search.
 * Utilizes Constant Memory for m[0..8] to minimize register pressure.
 */
__global__ __launch_bounds__(512, 2)
void blake2b_search_kernel(
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint32_t* out_found_nonces,
    uint32_t* out_count
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_nonces) return;

    uint32_t nonce = start_nonce + tid;

    // 1. Load precomputed midstate (state after Round 0 column step)
    uint64_t v0  = d_midstate.v[0];
    uint64_t v1  = d_midstate.v[1];
    uint64_t v2  = d_midstate.v[2];
    uint64_t v3  = d_midstate.v[3];
    uint64_t v4  = d_midstate.v[4];
    uint64_t v5  = d_midstate.v[5];
    uint64_t v6  = d_midstate.v[6];
    uint64_t v7  = d_midstate.v[7];
    uint64_t v8  = d_midstate.v[8];
    uint64_t v9  = d_midstate.v[9];
    uint64_t v10 = d_midstate.v[10];
    uint64_t v11 = d_midstate.v[11];
    uint64_t v12 = d_midstate.v[12];
    uint64_t v13 = d_midstate.v[13];
    uint64_t v14 = d_midstate.v[14];
    uint64_t v15 = d_midstate.v[15];

    // 2. Variable message word m9 (nBits [32-bit] + Nonce [32-bit])
    uint64_t m9 = ((uint64_t)nonce << 32) | (uint64_t)d_midstate.nbits;

    /* =====================================================================
     * Round 0: Only the 1 dynamic diagonal step (remaining 7 steps are in midstate!)
     * ===================================================================== */
    G(v0, v5, v10, v15, d_midstate.m[8], m9);

    /* =====================================================================
     * Round 1 (Sigma: 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8,  v12);
    G(v1, v5, v9,  v13, d_midstate.m[4], d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, m9);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8,  v13, d_midstate.m[7]);
    G(v3, v4, v9,  v14, d_midstate.m[5], d_midstate.m[3]);

    /* =====================================================================
     * Round 2 (Sigma: 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4)
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[8]);
    G_ZERO_X(v1, v5, v9,  v13, d_midstate.m[0]);
    G(v2, v6, v10, v14, d_midstate.m[5], d_midstate.m[2]);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G(v1, v6, v11, v12, d_midstate.m[3], d_midstate.m[6]);
    G(v2, v7, v8,  v13, d_midstate.m[7], d_midstate.m[1]);
    G(v3, v4, v9,  v14, m9, d_midstate.m[4]);

    /* =====================================================================
     * Round 3 (Sigma: 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8)
     * ===================================================================== */
    G(v0, v4, v8,  v12, d_midstate.m[7], m9);
    G(v1, v5, v9,  v13, d_midstate.m[3], d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G(v0, v5, v10, v15, d_midstate.m[2], d_midstate.m[6]);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[5]);
    G(v2, v7, v8,  v13, d_midstate.m[4], d_midstate.m[0]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 4 (Sigma: 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13)
     * ===================================================================== */
    G(v0, v4, v8,  v12, m9, d_midstate.m[0]);
    G(v1, v5, v9,  v13, d_midstate.m[5], d_midstate.m[7]);
    G(v2, v6, v10, v14, d_midstate.m[2], d_midstate.m[4]);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[1]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G(v2, v7, v8,  v13, d_midstate.m[6], d_midstate.m[8]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[3]);

    /* =====================================================================
     * Round 5 (Sigma: 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9)
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8,  v12, d_midstate.m[2]);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[6]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[0]);
    G(v3, v7, v11, v15, d_midstate.m[8], d_midstate.m[3]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[4]);
    G(v1, v6, v11, v12, d_midstate.m[7], d_midstate.m[5]);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G(v3, v4, v9,  v14, d_midstate.m[1], m9);

    /* =====================================================================
     * Round 6 (Sigma: 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11)
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[5]);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_Y(v3, v7, v11, v15, d_midstate.m[4]);

    G(v0, v5, v10, v15, d_midstate.m[0], d_midstate.m[7]);
    G(v1, v6, v11, v12, d_midstate.m[6], d_midstate.m[3]);
    G(v2, v7, v8,  v13, m9, d_midstate.m[2]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 7 (Sigma: 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10)
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8,  v12);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[7]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[1]);
    G(v3, v7, v11, v15, d_midstate.m[3], m9);

    G(v0, v5, v10, v15, d_midstate.m[5], d_midstate.m[0]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[4]);
    G(v2, v7, v8,  v13, d_midstate.m[8], d_midstate.m[6]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[2]);

    /* =====================================================================
     * Round 8 (Sigma: 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5)
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8,  v12, d_midstate.m[6]);
    G_ZERO_X(v1, v5, v9,  v13, m9);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[3]);
    G(v3, v7, v11, v15, d_midstate.m[0], d_midstate.m[8]);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[2]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[7]);
    G(v2, v7, v8,  v13, d_midstate.m[1], d_midstate.m[4]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[5]);

    /* =====================================================================
     * Round 9 (Sigma: 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0)
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[2]);
    G(v1, v5, v9,  v13, d_midstate.m[8], d_midstate.m[4]);
    G(v2, v6, v10, v14, d_midstate.m[7], d_midstate.m[6]);
    G(v3, v7, v11, v15, d_midstate.m[1], d_midstate.m[5]);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G_ZERO_Y(v1, v6, v11, v12, m9);
    G_ZERO_Y(v2, v7, v8,  v13, d_midstate.m[3]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[0]);

    /* =====================================================================
     * Round 10 (Sigma = Round 0)
     * ===================================================================== */
    G(v0, v4, v8,  v12, d_midstate.m[0], d_midstate.m[1]);
    G(v1, v5, v9,  v13, d_midstate.m[2], d_midstate.m[3]);
    G(v2, v6, v10, v14, d_midstate.m[4], d_midstate.m[5]);
    G(v3, v7, v11, v15, d_midstate.m[6], d_midstate.m[7]);

    G(v0, v5, v10, v15, d_midstate.m[8], m9);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G_ZERO_BOTH(v3, v4, v9,  v14);

    /* =====================================================================
     * Round 11 (Sigma = Round 1)
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8,  v12);
    G(v1, v5, v9,  v13, d_midstate.m[4], d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, m9);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8,  v13, d_midstate.m[7]);
    G(v3, v4, v9,  v14, d_midstate.m[5], d_midstate.m[3]);

    /* =====================================================================
     * Finalization & Target Comparison (PTX lop3.lut 3-way XOR + byte perm)
     * ===================================================================== */
    uint32_t v3_lo = (uint32_t)v3,   v3_hi = (uint32_t)(v3 >> 32);
    uint32_t v11_lo = (uint32_t)v11, v11_hi = (uint32_t)(v11 >> 32);

    uint32_t h3_lo = xor3_b32((uint32_t)BLAKE2B_256_INIT[3], v3_lo, v11_lo);
    uint32_t h3_hi = xor3_b32((uint32_t)(BLAKE2B_256_INIT[3] >> 32), v3_hi, v11_hi);

    uint32_t r_hi = __byte_perm(h3_lo, 0, 0x0123);
    uint32_t r_lo = __byte_perm(h3_hi, 0, 0x0123);
    uint64_t h3_be = ((uint64_t)r_hi << 32) | r_lo;

    if (h3_be <= target_diff) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < MAX_FOUND_NONCES) {
            out_found_nonces[idx] = nonce;
        }
    }
}

/**
 * @brief Loads precomputed midstate into GPU Constant Memory.
 */
extern "C" cudaError_t blake2b_set_midstate_cuda(const blake2b_midstate_t* host_midstate) {
    return cudaMemcpyToSymbol(d_midstate, host_midstate, sizeof(blake2b_midstate_t));
}

/**
 * @brief Launches optimized Blake2b search kernel.
 */
extern "C" cudaError_t blake2b_launch_kernel(
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint32_t* d_found_nonces,
    uint32_t* d_found_count,
    uint32_t block_size,
    cudaStream_t stream
) {
    uint32_t grid_size = (num_nonces + block_size - 1) / block_size;
    blake2b_search_kernel<<<grid_size, block_size, 0, stream>>>(
        start_nonce,
        num_nonces,
        target_diff,
        d_found_nonces,
        d_found_count
    );
    return cudaGetLastError();
}

/**
 * @brief Kernel for single hash calculation (for verification tests).
 */
__global__ void blake2b_single_hash_kernel(uint32_t nonce, uint64_t* out_h) {
    uint64_t v0  = d_midstate.v[0];
    uint64_t v1  = d_midstate.v[1];
    uint64_t v2  = d_midstate.v[2];
    uint64_t v3  = d_midstate.v[3];
    uint64_t v4  = d_midstate.v[4];
    uint64_t v5  = d_midstate.v[5];
    uint64_t v6  = d_midstate.v[6];
    uint64_t v7  = d_midstate.v[7];
    uint64_t v8  = d_midstate.v[8];
    uint64_t v9  = d_midstate.v[9];
    uint64_t v10 = d_midstate.v[10];
    uint64_t v11 = d_midstate.v[11];
    uint64_t v12 = d_midstate.v[12];
    uint64_t v13 = d_midstate.v[13];
    uint64_t v14 = d_midstate.v[14];
    uint64_t v15 = d_midstate.v[15];

    uint64_t m9 = ((uint64_t)nonce << 32) | (uint64_t)d_midstate.nbits;

    G(v0, v5, v10, v15, d_midstate.m[8], m9);

    G_ZERO_BOTH(v0, v4, v8,  v12);
    G(v1, v5, v9,  v13, d_midstate.m[4], d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, m9);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);
    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8,  v13, d_midstate.m[7]);
    G(v3, v4, v9,  v14, d_midstate.m[5], d_midstate.m[3]);

    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[8]);
    G_ZERO_X(v1, v5, v9,  v13, d_midstate.m[0]);
    G(v2, v6, v10, v14, d_midstate.m[5], d_midstate.m[2]);
    G_ZERO_BOTH(v3, v7, v11, v15);
    G_ZERO_BOTH(v0, v5, v10, v15);
    G(v1, v6, v11, v12, d_midstate.m[3], d_midstate.m[6]);
    G(v2, v7, v8,  v13, d_midstate.m[7], d_midstate.m[1]);
    G(v3, v4, v9,  v14, m9, d_midstate.m[4]);

    G(v0, v4, v8,  v12, d_midstate.m[7], m9);
    G(v1, v5, v9,  v13, d_midstate.m[3], d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_BOTH(v3, v7, v11, v15);
    G(v0, v5, v10, v15, d_midstate.m[2], d_midstate.m[6]);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[5]);
    G(v2, v7, v8,  v13, d_midstate.m[4], d_midstate.m[0]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[8]);

    G(v0, v4, v8,  v12, m9, d_midstate.m[0]);
    G(v1, v5, v9,  v13, d_midstate.m[5], d_midstate.m[7]);
    G(v2, v6, v10, v14, d_midstate.m[2], d_midstate.m[4]);
    G_ZERO_BOTH(v3, v7, v11, v15);
    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[1]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G(v2, v7, v8,  v13, d_midstate.m[6], d_midstate.m[8]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[3]);

    G_ZERO_Y(v0, v4, v8,  v12, d_midstate.m[2]);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[6]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[0]);
    G(v3, v7, v11, v15, d_midstate.m[8], d_midstate.m[3]);
    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[4]);
    G(v1, v6, v11, v12, d_midstate.m[7], d_midstate.m[5]);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G(v3, v4, v9,  v14, d_midstate.m[1], m9);

    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[5]);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_Y(v3, v7, v11, v15, d_midstate.m[4]);
    G(v0, v5, v10, v15, d_midstate.m[0], d_midstate.m[7]);
    G(v1, v6, v11, v12, d_midstate.m[6], d_midstate.m[3]);
    G(v2, v7, v8,  v13, m9, d_midstate.m[2]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[8]);

    G_ZERO_BOTH(v0, v4, v8,  v12);
    G_ZERO_Y(v1, v5, v9,  v13, d_midstate.m[7]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[1]);
    G(v3, v7, v11, v15, d_midstate.m[3], m9);
    G(v0, v5, v10, v15, d_midstate.m[5], d_midstate.m[0]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[4]);
    G(v2, v7, v8,  v13, d_midstate.m[8], d_midstate.m[6]);
    G_ZERO_Y(v3, v4, v9,  v14, d_midstate.m[2]);

    G_ZERO_Y(v0, v4, v8,  v12, d_midstate.m[6]);
    G_ZERO_X(v1, v5, v9,  v13, m9);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[3]);
    G(v3, v7, v11, v15, d_midstate.m[0], d_midstate.m[8]);
    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[2]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[7]);
    G(v2, v7, v8,  v13, d_midstate.m[1], d_midstate.m[4]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[5]);

    G_ZERO_X(v0, v4, v8,  v12, d_midstate.m[2]);
    G(v1, v5, v9,  v13, d_midstate.m[8], d_midstate.m[4]);
    G(v2, v6, v10, v14, d_midstate.m[7], d_midstate.m[6]);
    G(v3, v7, v11, v15, d_midstate.m[1], d_midstate.m[5]);
    G_ZERO_BOTH(v0, v5, v10, v15);
    G_ZERO_Y(v1, v6, v11, v12, m9);
    G_ZERO_Y(v2, v7, v8,  v13, d_midstate.m[3]);
    G_ZERO_X(v3, v4, v9,  v14, d_midstate.m[0]);

    G(v0, v4, v8,  v12, d_midstate.m[0], d_midstate.m[1]);
    G(v1, v5, v9,  v13, d_midstate.m[2], d_midstate.m[3]);
    G(v2, v6, v10, v14, d_midstate.m[4], d_midstate.m[5]);
    G(v3, v7, v11, v15, d_midstate.m[6], d_midstate.m[7]);
    G(v0, v5, v10, v15, d_midstate.m[8], m9);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G_ZERO_BOTH(v3, v4, v9,  v14);

    G_ZERO_BOTH(v0, v4, v8,  v12);
    G(v1, v5, v9,  v13, d_midstate.m[4], d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, m9);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);
    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8,  v13, d_midstate.m[7]);
    G(v3, v4, v9,  v14, d_midstate.m[5], d_midstate.m[3]);

    out_h[0] = xor3_b64(BLAKE2B_256_INIT[0], v0, v8);
    out_h[1] = xor3_b64(BLAKE2B_256_INIT[1], v1, v9);
    out_h[2] = xor3_b64(BLAKE2B_256_INIT[2], v2, v10);
    out_h[3] = xor3_b64(BLAKE2B_256_INIT[3], v3, v11);
}

extern "C" cudaError_t blake2b_compute_single_hash_gpu(uint32_t nonce, uint64_t out_h[4]) {
    uint64_t* d_h;
    cudaMalloc(&d_h, 4 * sizeof(uint64_t));
    blake2b_single_hash_kernel<<<1, 1>>>(nonce, d_h);
    cudaDeviceSynchronize();
    cudaMemcpy(out_h, d_h, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    return cudaGetLastError();
}
