/**
 * @file benchmark.cu
 * @brief Benchmark & parameter tuning (block-size sweeper & multi-stream pipelining) for Blake2bCudaMiner.
 */

#include "blake2b_cuda.cuh"
#include "blake2b_host.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>

extern "C" cudaError_t blake2b_set_midstate_cuda(const blake2b_midstate_t* host_midstate);
extern "C" cudaError_t blake2b_launch_kernel(
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint32_t* d_found_nonces,
    uint32_t* d_found_count,
    uint32_t block_size,
    cudaStream_t stream
);

void run_benchmark_for_block_size(uint32_t block_size, uint32_t* d_found, uint32_t* d_cnt) {
    const uint32_t batch_size = 64 * 1024 * 1024; // 67,108,864 nonces
    const int iterations = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    blake2b_launch_kernel(0, 1024 * 1024, 0x0000000000000000ULL, d_found, d_cnt, block_size, 0);
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    float min_ms = 99999.0f;

    for (int i = 0; i < iterations; ++i) {
        uint32_t start_nonce = i * batch_size;
        cudaEventRecord(start);
        blake2b_launch_kernel(start_nonce, batch_size, 0x0000000000000000ULL, d_found, d_cnt, block_size, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    double best_hashrate = ((double)batch_size / (min_ms / 1000.0)) / 1e6;
    double avg_hashrate = ((double)batch_size / ((total_ms / iterations) / 1000.0)) / 1e6;

    std::cout << "  • Block size " << std::setw(3) << block_size << ": "
              << "Best: " << std::fixed << std::setprecision(2) << min_ms << " ms (" << best_hashrate / 1000.0 << " GH/s)"
              << " | Avg: " << avg_hashrate / 1000.0 << " GH/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_pipelining_benchmark(uint32_t block_size) {
    const uint32_t batch_size = 64 * 1024 * 1024; // 67,108,864 nonces
    const int total_batches = 10;
    const uint64_t total_nonces = (uint64_t)batch_size * total_batches;

    std::cout << "\n 🔄 STREAM PIPELINING BENCHMARK (" << total_batches << " batches, "
              << total_nonces / 1000000 << "M nonces, block size " << block_size << "):" << std::endl;

    // 1. Synchronous Single-Stream Baseline
    uint32_t* d_found_nonces;
    uint32_t* d_found_count;
    cudaMalloc(&d_found_nonces, 16 * sizeof(uint32_t));
    cudaMalloc(&d_found_count, sizeof(uint32_t));
    uint32_t sync_count = 0;

    // Warmup
    blake2b_launch_kernel(0, 1024 * 1024, 0ULL, d_found_nonces, d_found_count, block_size, 0);
    cudaDeviceSynchronize();

    auto t_sync_start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < total_batches; ++b) {
        cudaMemset(d_found_count, 0, sizeof(uint32_t));
        blake2b_launch_kernel((uint32_t)b * batch_size, batch_size, 0ULL, d_found_nonces, d_found_count, block_size, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(&sync_count, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    auto t_sync_end = std::chrono::high_resolution_clock::now();
    double sync_sec = std::chrono::duration<double>(t_sync_end - t_sync_start).count();
    double sync_ghs = ((double)total_nonces / sync_sec) / 1e9;

    cudaFree(d_found_nonces);
    cudaFree(d_found_count);

    // 2. Asynchronous Dual-Stream (Double-Buffering)
    cudaStream_t streams[2];
    uint32_t* d_nonces[2];
    uint32_t* d_count[2];
    uint32_t* h_count[2];
    uint32_t* h_nonces[2];

    for (int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        cudaMalloc(&d_nonces[i], 16 * sizeof(uint32_t));
        cudaMalloc(&d_count[i], sizeof(uint32_t));
        cudaMallocHost(&h_nonces[i], 16 * sizeof(uint32_t));
        cudaMallocHost(&h_count[i], sizeof(uint32_t));
    }

    auto t_pipe_start = std::chrono::high_resolution_clock::now();
    int active = 0;
    bool in_flight[2] = {false, false};

    for (int b = 0; b < total_batches; ++b) {
        if (in_flight[active]) {
            cudaStreamSynchronize(streams[active]);
            in_flight[active] = false;
        }

        cudaMemsetAsync(d_count[active], 0, sizeof(uint32_t), streams[active]);
        blake2b_launch_kernel((uint32_t)b * batch_size, batch_size, 0ULL, d_nonces[active], d_count[active], block_size, streams[active]);
        cudaMemcpyAsync(h_count[active], d_count[active], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[active]);
        in_flight[active] = true;

        active = 1 - active;
    }

    // Flush remaining in-flight stream
    for (int i = 0; i < 2; ++i) {
        if (in_flight[i]) {
            cudaStreamSynchronize(streams[i]);
            in_flight[i] = false;
        }
    }
    auto t_pipe_end = std::chrono::high_resolution_clock::now();
    double pipe_sec = std::chrono::duration<double>(t_pipe_end - t_pipe_start).count();
    double pipe_ghs = ((double)total_nonces / pipe_sec) / 1e9;

    // Cleanup
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_nonces[i]);
        cudaFree(d_count[i]);
        cudaFreeHost(h_nonces[i]);
        cudaFreeHost(h_count[i]);
        cudaStreamDestroy(streams[i]);
    }

    double speedup_pct = ((pipe_ghs - sync_ghs) / sync_ghs) * 100.0;

    std::cout << "  • Synchronous (1 Stream):   " << std::fixed << std::setprecision(3)
              << sync_sec * 1000.0 << " ms (" << sync_ghs << " GH/s)" << std::endl;
    std::cout << "  • Asynchronous (2 Streams): " << std::fixed << std::setprecision(3)
              << pipe_sec * 1000.0 << " ms (" << pipe_ghs << " GH/s)" << std::endl;
    std::cout << "  • Speedup / Throughput Gain: +" << std::fixed << std::setprecision(2)
              << speedup_pct << "% (+" << (pipe_ghs - sync_ghs) * 1000.0 << " MH/s)" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " 🚀 BLAKE2BCUDAMINER - BLOCK-SIZE & GRID TUNING" << std::endl;
    std::cout << "============================================================" << std::endl;

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "  • GPU:                   " << prop.name << std::endl;
    std::cout << "  • SMs (Multiprocessors): " << prop.multiProcessorCount << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    uint8_t test_header[80];
    for (int i = 0; i < 80; ++i) test_header[i] = (uint8_t)(i * 3 + 7);
    uint32_t test_nbits = 0x1e00ffff;

    blake2b_midstate_t midstate;
    blake2b_precompute_midstate(test_header, test_nbits, &midstate);
    blake2b_set_midstate_cuda(&midstate);

    uint32_t* d_found_nonces;
    uint32_t* d_found_count;
    cudaMalloc(&d_found_nonces, 16 * sizeof(uint32_t));
    cudaMalloc(&d_found_count, sizeof(uint32_t));
    cudaMemset(d_found_count, 0, sizeof(uint32_t));

    std::cout << " ⚡ Testing thread block sizes (threads/block)..." << std::endl;
    std::vector<uint32_t> block_sizes = {64, 128, 256, 512};
    for (uint32_t bs : block_sizes) {
        run_benchmark_for_block_size(bs, d_found_nonces, d_found_count);
    }

    std::cout << "============================================================" << std::endl;

    cudaFree(d_found_nonces);
    cudaFree(d_found_count);

    // Multi-Stream Pipelining Benchmark
    run_pipelining_benchmark(512);

    std::cout << "============================================================" << std::endl;
    return 0;
}
