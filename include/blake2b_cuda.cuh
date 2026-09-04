#ifndef BLAKE2B_CUDA_CUH
#define BLAKE2B_CUDA_CUH

/**
 * @file blake2b_cuda.cuh
 * @brief Highly optimized CUDA definitions and PTX inline assembly for Blake2b (Bitcoin Knots PoW).
 */

#include <cstdint>
#include <cuda_runtime.h>

// Blake2b Initialization Vector (IV)
__constant__ const uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// Initialized state for Blake2b-256 (IV[0] ^ 0x01010020)
__constant__ const uint64_t BLAKE2B_256_INIT[8] = {
    0x6a09e667f3bcc908ULL ^ 0x01010020ULL,
    0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,
    0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL
};

/**
 * @struct blake2b_midstate_t
 * @brief Precomputed state for 80-byte Bitcoin header.
 * Contains working state v[0..15] after Round 0 column step as well as m[0..8] and nBits.
 */
struct blake2b_midstate_t {
    uint64_t v[16];      // Precomputed internal working state after R0 column static steps
    uint64_t m[10];      // Message words m[0..3] and m[5..9] (m[4] high 32-bit holds nonce2)
    uint32_t nonce2;     // High 32-bit of m[4]
    uint32_t target_hi;  // Highest 32 bits of search target
    uint64_t target64;   // 64-bit compact target for fast preliminary check
};

#ifdef __CUDACC__

/* =========================================================================
 * Hardware-accelerated PTX 64-bit rotations (funnel shifts)
 * ========================================================================= */

/**
 * @brief 64-bit right rotation by 32 bits.
 */
__device__ __forceinline__ uint64_t rotr64_32(uint64_t x) {
    return (x >> 32) | (x << 32);
}

/**
 * @brief 64-bit right rotation by 24 bits using CUDA funnel shift.
 */
__device__ __forceinline__ uint64_t rotr64_24(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t r_lo = __funnelshift_r(lo, hi, 24);
    uint32_t r_hi = __funnelshift_r(hi, lo, 24);
    return ((uint64_t)r_hi << 32) | r_lo;
}

/**
 * @brief 64-bit right rotation by 16 bits using CUDA funnel shift.
 */
__device__ __forceinline__ uint64_t rotr64_16(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t r_lo = __funnelshift_r(lo, hi, 16);
    uint32_t r_hi = __funnelshift_r(hi, lo, 16);
    return ((uint64_t)r_hi << 32) | r_lo;
}

/**
 * @brief 64-bit right rotation by 63 bits (equivalent to left rotation by 1 bit).
 */
__device__ __forceinline__ uint64_t rotr64_63(uint64_t x) {
    return (x >> 63) | (x << 1);
}

/**
 * @brief 64-bit byte swap (little-endian to big-endian) in device code.
 */
__device__ __forceinline__ uint64_t bswap64_device(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t r_lo = __byte_perm(hi, 0, 0x0123);
    uint32_t r_hi = __byte_perm(lo, 0, 0x0123);
    return ((uint64_t)r_hi << 32) | r_lo;
}

/**
 * @brief Fuses a 3-input XOR (a ^ b ^ c) into a single 1-cycle PTX instruction.
 * Uses NVIDIA lop3.lut with truth table 0x96.
 */
__device__ __forceinline__ uint32_t xor3_b32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

/**
 * @brief 64-bit 3-input XOR using two 1-cycle lop3.b32 instructions.
 */
__device__ __forceinline__ uint64_t xor3_b64(uint64_t a, uint64_t b, uint64_t c) {
    uint32_t a_lo = (uint32_t)a, a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b, b_hi = (uint32_t)(b >> 32);
    uint32_t c_lo = (uint32_t)c, c_hi = (uint32_t)(c >> 32);
    uint32_t r_lo = xor3_b32(a_lo, b_lo, c_lo);
    uint32_t r_hi = xor3_b32(a_hi, b_hi, c_hi);
    return ((uint64_t)r_hi << 32) | r_lo;
}

/* =========================================================================
 * Blake2b G-function macros with Zero-Folding
 * ========================================================================= */

// Standard G function (both message parameters present)
#define G(a, b, c, d, mx, my) \
    do { \
        a = a + b + (mx); \
        d = rotr64_32(d ^ a); \
        c = c + d; \
        b = rotr64_24(b ^ c); \
        a = a + b + (my); \
        d = rotr64_16(d ^ a); \
        c = c + d; \
        b = rotr64_63(b ^ c); \
    } while(0)

// Zero-Folding: mx == 0 (eliminates 1 64-bit addition)
#define G_ZERO_X(a, b, c, d, my) \
    do { \
        a = a + b; \
        d = rotr64_32(d ^ a); \
        c = c + d; \
        b = rotr64_24(b ^ c); \
        a = a + b + (my); \
        d = rotr64_16(d ^ a); \
        c = c + d; \
        b = rotr64_63(b ^ c); \
    } while(0)

// Zero-Folding: my == 0 (eliminates 1 64-bit addition)
#define G_ZERO_Y(a, b, c, d, mx) \
    do { \
        a = a + b + (mx); \
        d = rotr64_32(d ^ a); \
        c = c + d; \
        b = rotr64_24(b ^ c); \
        a = a + b; \
        d = rotr64_16(d ^ a); \
        c = c + d; \
        b = rotr64_63(b ^ c); \
    } while(0)

// Zero-Folding: mx == 0 AND my == 0 (eliminates 2 64-bit additions)
#define G_ZERO_BOTH(a, b, c, d) \
    do { \
        a = a + b; \
        d = rotr64_32(d ^ a); \
        c = c + d; \
        b = rotr64_24(b ^ c); \
        a = a + b; \
        d = rotr64_16(d ^ a); \
        c = c + d; \
        b = rotr64_63(b ^ c); \
    } while(0)

#endif // __CUDACC__

#endif // BLAKE2B_CUDA_CUH
