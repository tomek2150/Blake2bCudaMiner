#ifndef BLAKE2B_HOST_H
#define BLAKE2B_HOST_H

/**
 * @file blake2b_host.h
 * @brief Host-side midstate precomputation and reference hashing for Blake2bCudaMiner.
 */

#include "blake2b_cuda.cuh"
#include <cstdint>
#include <vector>

/**
 * @brief Precomputes midstate for an 80-byte Bitcoin header.
 * 
 * @param header_80 Pointer to the 80 bytes of the block header.
 * @param target_diff_bits nBits of current difficulty target.
 * @param out_midstate Output structure with precomputed midstate.
 */
void blake2b_precompute_midstate(const uint8_t header_80[80], uint32_t target_diff_bits, blake2b_midstate_t* out_midstate);

/**
 * @brief CPU reference implementation of Blake2b-256 over 80 bytes.
 * 
 * @param header_80 80-byte block header.
 * @param out_hash_32 Output buffer for the 32-byte hash.
 */
void blake2b_256_cpu_reference(const uint8_t header_80[80], uint8_t out_hash_32[32]);

#endif // BLAKE2B_HOST_H
