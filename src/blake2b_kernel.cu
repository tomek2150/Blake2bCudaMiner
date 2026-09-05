/**
 * @file blake2b_kernel.cu
 * @brief Highly optimized CUDA kernel for Blake2b PoW with Constant Memory message broadcast and PTX.
 */

#include "blake2b_cuda.cuh"

// Maximum matches per kernel launch
#define MAX_FOUND_NONCES 16

__constant__ blake2b_midstate_t d_midstate;

/**
 * @brief CUDA kernel for parallel Blake2b nonce search (Bitcoin Knots Profile 0).
 * Utilizes Constant Memory for static message words to minimize register pressure.
 */
__global__ __launch_bounds__(512, 2)
void blake2b_search_kernel(
    uint64_t start_nonce64,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint64_t* out_found_nonces,
    uint32_t* out_count
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_nonces) return;

    uint64_t full_nonce = start_nonce64 + tid;

    // 1. Load precomputed midstate (state after Round 0 column static steps)
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

    // 2. Variable message word m4 (Nonce [low 32-bit] + nonce2 [high 32-bit])
    uint64_t m4 = full_nonce;

    /* =====================================================================
     * Round 0: Complete Column 2 and Diagonal steps on GPU
     * ===================================================================== */
    G(v2, v6, v10, v14, m4, d_midstate.m[5]);
    G(v0, v5, v10, v15, d_midstate.m[8], d_midstate.m[9]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G_ZERO_BOTH(v3, v4, v9,  v14);

    /* =====================================================================
     * Round 1
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G(v1, v5, v9, v13, m4, d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[9]);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8, v13, d_midstate.m[7]);
    G(v3, v4, v9, v14, d_midstate.m[5], d_midstate.m[3]);

    /* =====================================================================
     * Round 2
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[8]);
    G_ZERO_X(v1, v5, v9, v13, d_midstate.m[0]);
    G(v2, v6, v10, v14, d_midstate.m[5], d_midstate.m[2]);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G(v1, v6, v11, v12, d_midstate.m[3], d_midstate.m[6]);
    G(v2, v7, v8, v13, d_midstate.m[7], d_midstate.m[1]);
    G(v3, v4, v9, v14, d_midstate.m[9], m4);

    /* =====================================================================
     * Round 3
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[7], d_midstate.m[9]);
    G(v1, v5, v9, v13, d_midstate.m[3], d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G(v0, v5, v10, v15, d_midstate.m[2], d_midstate.m[6]);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[5]);
    G(v2, v7, v8, v13, m4, d_midstate.m[0]);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 4
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[9], d_midstate.m[0]);
    G(v1, v5, v9, v13, d_midstate.m[5], d_midstate.m[7]);
    G(v2, v6, v10, v14, d_midstate.m[2], m4);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[1]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G(v2, v7, v8, v13, d_midstate.m[6], d_midstate.m[8]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[3]);

    /* =====================================================================
     * Round 5
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8, v12, d_midstate.m[2]);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[6]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[0]);
    G(v3, v7, v11, v15, d_midstate.m[8], d_midstate.m[3]);

    G_ZERO_Y(v0, v5, v10, v15, m4);
    G(v1, v6, v11, v12, d_midstate.m[7], d_midstate.m[5]);
    G_ZERO_BOTH(v2, v7, v8, v13);
    G(v3, v4, v9, v14, d_midstate.m[1], d_midstate.m[9]);

    /* =====================================================================
     * Round 6
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[5]);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_Y(v3, v7, v11, v15, m4);

    G(v0, v5, v10, v15, d_midstate.m[0], d_midstate.m[7]);
    G(v1, v6, v11, v12, d_midstate.m[6], d_midstate.m[3]);
    G(v2, v7, v8, v13, d_midstate.m[9], d_midstate.m[2]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 7
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[7]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[1]);
    G(v3, v7, v11, v15, d_midstate.m[3], d_midstate.m[9]);

    G(v0, v5, v10, v15, d_midstate.m[5], d_midstate.m[0]);
    G_ZERO_X(v1, v6, v11, v12, m4);
    G(v2, v7, v8, v13, d_midstate.m[8], d_midstate.m[6]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[2]);

    /* =====================================================================
     * Round 8
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8, v12, d_midstate.m[6]);
    G_ZERO_X(v1, v5, v9, v13, d_midstate.m[9]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[3]);
    G(v3, v7, v11, v15, d_midstate.m[0], d_midstate.m[8]);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[2]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[7]);
    G(v2, v7, v8, v13, d_midstate.m[1], m4);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[5]);

    /* =====================================================================
     * Round 9
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[2]);
    G(v1, v5, v9, v13, d_midstate.m[8], m4);
    G(v2, v6, v10, v14, d_midstate.m[7], d_midstate.m[6]);
    G(v3, v7, v11, v15, d_midstate.m[1], d_midstate.m[5]);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[9]);
    G_ZERO_Y(v2, v7, v8, v13, d_midstate.m[3]);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[0]);

    /* =====================================================================
     * Round 10
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[0], d_midstate.m[1]);
    G(v1, v5, v9, v13, d_midstate.m[2], d_midstate.m[3]);
    G(v2, v6, v10, v14, m4, d_midstate.m[5]);
    G(v3, v7, v11, v15, d_midstate.m[6], d_midstate.m[7]);

    G(v0, v5, v10, v15, d_midstate.m[8], d_midstate.m[9]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8, v13);
    G_ZERO_BOTH(v3, v4, v9, v14);

    /* =====================================================================
     * Round 11
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G(v1, v5, v9, v13, m4, d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[9]);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8, v13, d_midstate.m[7]);
    G(v3, v4, v9, v14, d_midstate.m[5], d_midstate.m[3]);

    /* =====================================================================
     * Finalization & Target Comparison (PTX lop3.lut 3-way XOR + byte perm)
     * ===================================================================== */
    uint32_t v0_lo = (uint32_t)v0,   v0_hi = (uint32_t)(v0 >> 32);
    uint32_t v8_lo = (uint32_t)v8,   v8_hi = (uint32_t)(v8 >> 32);

    uint32_t h0_lo = xor3_b32((uint32_t)BLAKE2B_256_INIT[0], v0_lo, v8_lo);
    uint32_t h0_hi = xor3_b32((uint32_t)(BLAKE2B_256_INIT[0] >> 32), v0_hi, v8_hi);

    uint32_t r_hi = __byte_perm(h0_lo, 0, 0x0123);
    uint32_t r_lo = __byte_perm(h0_hi, 0, 0x0123);
    uint64_t h0_be = ((uint64_t)r_hi << 32) | r_lo;

    if (h0_be <= target_diff) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < MAX_FOUND_NONCES) {
            out_found_nonces[idx] = full_nonce;
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
    uint64_t start_nonce64,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint64_t* d_found_nonces,
    uint32_t* d_found_count,
    uint32_t block_size,
    cudaStream_t stream
) {
    uint32_t grid_size = (num_nonces + block_size - 1) / block_size;
    blake2b_search_kernel<<<grid_size, block_size, 0, stream>>>(
        start_nonce64,
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
__global__ void blake2b_single_hash_kernel(uint64_t full_nonce, uint64_t* out_h) {
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

    uint64_t m4 = full_nonce;

    /* =====================================================================
     * Round 0: Complete Column 2 and Diagonal steps on GPU
     * ===================================================================== */
    G(v2, v6, v10, v14, m4, d_midstate.m[5]);
    G(v0, v5, v10, v15, d_midstate.m[8], d_midstate.m[9]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8,  v13);
    G_ZERO_BOTH(v3, v4, v9,  v14);

    /* =====================================================================
     * Round 1
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G(v1, v5, v9, v13, m4, d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[9]);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8, v13, d_midstate.m[7]);
    G(v3, v4, v9, v14, d_midstate.m[5], d_midstate.m[3]);

    /* =====================================================================
     * Round 2
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[8]);
    G_ZERO_X(v1, v5, v9, v13, d_midstate.m[0]);
    G(v2, v6, v10, v14, d_midstate.m[5], d_midstate.m[2]);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G(v1, v6, v11, v12, d_midstate.m[3], d_midstate.m[6]);
    G(v2, v7, v8, v13, d_midstate.m[7], d_midstate.m[1]);
    G(v3, v4, v9, v14, d_midstate.m[9], m4);

    /* =====================================================================
     * Round 3
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[7], d_midstate.m[9]);
    G(v1, v5, v9, v13, d_midstate.m[3], d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G(v0, v5, v10, v15, d_midstate.m[2], d_midstate.m[6]);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[5]);
    G(v2, v7, v8, v13, m4, d_midstate.m[0]);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 4
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[9], d_midstate.m[0]);
    G(v1, v5, v9, v13, d_midstate.m[5], d_midstate.m[7]);
    G(v2, v6, v10, v14, d_midstate.m[2], m4);
    G_ZERO_BOTH(v3, v7, v11, v15);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[1]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G(v2, v7, v8, v13, d_midstate.m[6], d_midstate.m[8]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[3]);

    /* =====================================================================
     * Round 5
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8, v12, d_midstate.m[2]);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[6]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[0]);
    G(v3, v7, v11, v15, d_midstate.m[8], d_midstate.m[3]);

    G_ZERO_Y(v0, v5, v10, v15, m4);
    G(v1, v6, v11, v12, d_midstate.m[7], d_midstate.m[5]);
    G_ZERO_BOTH(v2, v7, v8, v13);
    G(v3, v4, v9, v14, d_midstate.m[1], d_midstate.m[9]);

    /* =====================================================================
     * Round 6
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[5]);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[1]);
    G_ZERO_BOTH(v2, v6, v10, v14);
    G_ZERO_Y(v3, v7, v11, v15, m4);

    G(v0, v5, v10, v15, d_midstate.m[0], d_midstate.m[7]);
    G(v1, v6, v11, v12, d_midstate.m[6], d_midstate.m[3]);
    G(v2, v7, v8, v13, d_midstate.m[9], d_midstate.m[2]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[8]);

    /* =====================================================================
     * Round 7
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G_ZERO_Y(v1, v5, v9, v13, d_midstate.m[7]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[1]);
    G(v3, v7, v11, v15, d_midstate.m[3], d_midstate.m[9]);

    G(v0, v5, v10, v15, d_midstate.m[5], d_midstate.m[0]);
    G_ZERO_X(v1, v6, v11, v12, m4);
    G(v2, v7, v8, v13, d_midstate.m[8], d_midstate.m[6]);
    G_ZERO_Y(v3, v4, v9, v14, d_midstate.m[2]);

    /* =====================================================================
     * Round 8
     * ===================================================================== */
    G_ZERO_Y(v0, v4, v8, v12, d_midstate.m[6]);
    G_ZERO_X(v1, v5, v9, v13, d_midstate.m[9]);
    G_ZERO_X(v2, v6, v10, v14, d_midstate.m[3]);
    G(v3, v7, v11, v15, d_midstate.m[0], d_midstate.m[8]);

    G_ZERO_X(v0, v5, v10, v15, d_midstate.m[2]);
    G_ZERO_X(v1, v6, v11, v12, d_midstate.m[7]);
    G(v2, v7, v8, v13, d_midstate.m[1], m4);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[5]);

    /* =====================================================================
     * Round 9
     * ===================================================================== */
    G_ZERO_X(v0, v4, v8, v12, d_midstate.m[2]);
    G(v1, v5, v9, v13, d_midstate.m[8], m4);
    G(v2, v6, v10, v14, d_midstate.m[7], d_midstate.m[6]);
    G(v3, v7, v11, v15, d_midstate.m[1], d_midstate.m[5]);

    G_ZERO_BOTH(v0, v5, v10, v15);
    G_ZERO_Y(v1, v6, v11, v12, d_midstate.m[9]);
    G_ZERO_Y(v2, v7, v8, v13, d_midstate.m[3]);
    G_ZERO_X(v3, v4, v9, v14, d_midstate.m[0]);

    /* =====================================================================
     * Round 10
     * ===================================================================== */
    G(v0, v4, v8, v12, d_midstate.m[0], d_midstate.m[1]);
    G(v1, v5, v9, v13, d_midstate.m[2], d_midstate.m[3]);
    G(v2, v6, v10, v14, m4, d_midstate.m[5]);
    G(v3, v7, v11, v15, d_midstate.m[6], d_midstate.m[7]);

    G(v0, v5, v10, v15, d_midstate.m[8], d_midstate.m[9]);
    G_ZERO_BOTH(v1, v6, v11, v12);
    G_ZERO_BOTH(v2, v7, v8, v13);
    G_ZERO_BOTH(v3, v4, v9, v14);

    /* =====================================================================
     * Round 11
     * ===================================================================== */
    G_ZERO_BOTH(v0, v4, v8, v12);
    G(v1, v5, v9, v13, m4, d_midstate.m[8]);
    G_ZERO_Y(v2, v6, v10, v14, d_midstate.m[9]);
    G_ZERO_X(v3, v7, v11, v15, d_midstate.m[6]);

    G_ZERO_Y(v0, v5, v10, v15, d_midstate.m[1]);
    G(v1, v6, v11, v12, d_midstate.m[0], d_midstate.m[2]);
    G_ZERO_X(v2, v7, v8, v13, d_midstate.m[7]);
    G(v3, v4, v9, v14, d_midstate.m[5], d_midstate.m[3]);

    out_h[0] = xor3_b64(BLAKE2B_256_INIT[0], v0, v8);
    out_h[1] = xor3_b64(BLAKE2B_256_INIT[1], v1, v9);
    out_h[2] = xor3_b64(BLAKE2B_256_INIT[2], v2, v10);
    out_h[3] = xor3_b64(BLAKE2B_256_INIT[3], v3, v11);
}

extern "C" cudaError_t blake2b_compute_single_hash_gpu(uint64_t full_nonce, uint64_t out_h[4]) {
    uint64_t* d_h;
    cudaMalloc(&d_h, 4 * sizeof(uint64_t));
    blake2b_single_hash_kernel<<<1, 1>>>(full_nonce, d_h);
    cudaDeviceSynchronize();
    cudaMemcpy(out_h, d_h, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    return cudaGetLastError();
}
