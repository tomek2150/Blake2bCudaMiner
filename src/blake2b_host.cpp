/**
 * @file blake2b_host.cpp
 * @brief Host midstate precomputation and CPU reference hashing for Blake2bCudaMiner.
 */

#include "blake2b_host.h"
#include <cstring>

// Blake2b permutation table (Sigma)
static const uint8_t SIGMA[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3}
};

static const uint64_t IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static inline uint64_t rotr64_cpu(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

#define G_CPU(v, a, b, c, d, mx, my) \
    do { \
        v[a] = v[a] + v[b] + (mx); \
        v[d] = rotr64_cpu(v[d] ^ v[a], 32); \
        v[c] = v[c] + v[d]; \
        v[b] = rotr64_cpu(v[b] ^ v[c], 24); \
        v[a] = v[a] + v[b] + (my); \
        v[d] = rotr64_cpu(v[d] ^ v[a], 16); \
        v[c] = v[c] + v[d]; \
        v[b] = rotr64_cpu(v[b] ^ v[c], 63); \
    } while(0)

void blake2b_precompute_midstate(const uint8_t header_80[80], uint32_t target_diff_bits, blake2b_midstate_t* out_midstate) {
    // 1. Load message words m[0..9] from the 80 bytes
    uint64_t m[16] = {0};
    std::memcpy(m, header_80, 80);

    // Copy static words m[0..9] to midstate (m[4] high 32-bit is nonce2)
    for (int i = 0; i < 10; ++i) {
        out_midstate->m[i] = m[i];
    }
    out_midstate->nonce2 = (uint32_t)(m[4] >> 32);

    // 2. Initialize working state v[0..15]
    uint64_t v[16];
    v[0] = IV[0] ^ 0x01010020ULL; // Blake2b-256 digest length (32 bytes)
    v[1] = IV[1];
    v[2] = IV[2];
    v[3] = IV[3];
    v[4] = IV[4];
    v[5] = IV[5];
    v[6] = IV[6];
    v[7] = IV[7];
    v[8]  = IV[0];
    v[9]  = IV[1];
    v[10] = IV[2];
    v[11] = IV[3];
    v[12] = IV[4] ^ 80ULL;                  // t0 = 80-byte header length
    v[13] = IV[5] ^ 0ULL;                   // t1 = 0
    v[14] = IV[6] ^ 0xFFFFFFFFFFFFFFFFULL;  // f0 = ~0 (final block)
    v[15] = IV[7] ^ 0ULL;                   // f1 = 0

    // 3. Precompute Round 0 static column steps (Column 0, 1, 3)
    G_CPU(v, 0, 4,  8, 12, m[0], m[1]);
    G_CPU(v, 1, 5,  9, 13, m[2], m[3]);
    // Column 2: G_CPU(v, 2, 6, 10, 14, m[4], m[5]) uses dynamic m[4] and is executed on GPU!
    G_CPU(v, 3, 7, 11, 15, m[6], m[7]);

    // Store precomputed state v[0..15]
    for (int i = 0; i < 16; ++i) {
        out_midstate->v[i] = v[i];
    }

    // Compact target (target high word)
    uint32_t exponent = target_diff_bits >> 24;
    uint32_t mantissa = target_diff_bits & 0x007FFFFF;
    if (exponent <= 3) {
        out_midstate->target64 = mantissa >> (8 * (3 - exponent));
    } else {
        out_midstate->target64 = (uint64_t)mantissa << (8 * (exponent - 3));
    }
    out_midstate->target_hi = (uint32_t)(out_midstate->target64 >> 32);
}

void blake2b_256_cpu_reference(const uint8_t header_80[80], uint8_t out_hash_32[32]) {
    uint64_t m[16] = {0};
    std::memcpy(m, header_80, 80);

    uint64_t h[8];
    h[0] = IV[0] ^ 0x01010020ULL;
    h[1] = IV[1];
    h[2] = IV[2];
    h[3] = IV[3];
    h[4] = IV[4];
    h[5] = IV[5];
    h[6] = IV[6];
    h[7] = IV[7];

    uint64_t v[16];
    for (int i = 0; i < 8; ++i) v[i] = h[i];
    v[8]  = IV[0];
    v[9]  = IV[1];
    v[10] = IV[2];
    v[11] = IV[3];
    v[12] = IV[4] ^ 80ULL;
    v[13] = IV[5] ^ 0ULL;
    v[14] = IV[6] ^ 0xFFFFFFFFFFFFFFFFULL;
    v[15] = IV[7] ^ 0ULL;

    for (int r = 0; r < 12; ++r) {
        // Columns
        G_CPU(v, 0, 4,  8, 12, m[SIGMA[r][0]],  m[SIGMA[r][1]]);
        G_CPU(v, 1, 5,  9, 13, m[SIGMA[r][2]],  m[SIGMA[r][3]]);
        G_CPU(v, 2, 6, 10, 14, m[SIGMA[r][4]],  m[SIGMA[r][5]]);
        G_CPU(v, 3, 7, 11, 15, m[SIGMA[r][6]],  m[SIGMA[r][7]]);
        // Diagonals
        G_CPU(v, 0, 5, 10, 15, m[SIGMA[r][8]],  m[SIGMA[r][9]]);
        G_CPU(v, 1, 6, 11, 12, m[SIGMA[r][10]], m[SIGMA[r][11]]);
        G_CPU(v, 2, 7,  8, 13, m[SIGMA[r][12]], m[SIGMA[r][13]]);
        G_CPU(v, 3, 4,  9, 14, m[SIGMA[r][14]], m[SIGMA[r][15]]);
    }

    uint64_t out_words[4];
    for (int i = 0; i < 4; ++i) {
        out_words[i] = h[i] ^ v[i] ^ v[i + 8];
    }
    std::memcpy(out_hash_32, out_words, 32);
}
