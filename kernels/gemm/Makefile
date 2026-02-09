NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGETS := bf16_gemm_naive bf16_gemm_tiled

all: $(TARGETS)

bf16_gemm_naive: bf16_gemm_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

bf16_gemm_tiled: bf16_gemm_tiled.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
