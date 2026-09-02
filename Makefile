# ==============================================================================
# Makefile for Blake2bCudaMiner: High-Efficiency Blake2b GPU Miner for Bitcoin Knots
# ==============================================================================

NVCC       := nvcc
CXX        := g++
CUDA_PATH  ?= /usr/local/cuda

# Compiler flags for maximum performance across all NVIDIA architectures (RTX 20xx to RTX 50xx)
NVCCFLAGS  := -O3 -std=c++17 -Iinclude -use_fast_math \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90 \
              -gencode arch=compute_90,code=compute_90 \
              -Xptxas -v,-O3 \
              -Xcompiler "-Wall,-Wextra,-O3,-march=native,-pthread"

CXXFLAGS   := -O3 -std=c++17 -Iinclude -Wall -Wextra -march=native -pthread
LDFLAGS    := -L$(CUDA_PATH)/lib64 -lcudart -lpthread

SRCDIR     := src
INCDIR     := include
TESTDIR    := tests
BUILDDIR   := build
BINDIR     := bin

# Targets
MINER_BIN  := $(BINDIR)/b2bcudaminer
BENCHMARK  := $(BINDIR)/benchmark
TEST_CORR  := $(BINDIR)/test_correctness

.PHONY: all clean dirs miner benchmark test

all: dirs miner benchmark test

dirs:
	@mkdir -p $(BUILDDIR) $(BINDIR)

# Compile Host C++
$(BUILDDIR)/blake2b_host.o: $(SRCDIR)/blake2b_host.cpp $(INCDIR)/blake2b_host.h $(INCDIR)/blake2b_cuda.cuh
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/stratum_client.o: $(SRCDIR)/stratum_client.cpp $(INCDIR)/stratum_client.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA Kernel
$(BUILDDIR)/blake2b_kernel.o: $(SRCDIR)/blake2b_kernel.cu $(INCDIR)/blake2b_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Standalone Miner Binary (Blake2bCudaMiner)
miner: dirs $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/stratum_client.o $(BUILDDIR)/blake2b_kernel.o
	$(NVCC) $(NVCCFLAGS) $(SRCDIR)/main.cpp $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/stratum_client.o $(BUILDDIR)/blake2b_kernel.o -o $(MINER_BIN) $(LDFLAGS) -lssl -lcrypto
	@echo "✅ Standalone miner built: $(MINER_BIN)"

# Benchmark Binary
benchmark: dirs $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/blake2b_kernel.o
	$(NVCC) $(NVCCFLAGS) $(TESTDIR)/benchmark.cu $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/blake2b_kernel.o -o $(BENCHMARK) $(LDFLAGS)

# Correctness Test Binary
test: dirs $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/blake2b_kernel.o
	$(NVCC) $(NVCCFLAGS) $(TESTDIR)/test_correctness.cu $(BUILDDIR)/blake2b_host.o $(BUILDDIR)/blake2b_kernel.o -o $(TEST_CORR) $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
