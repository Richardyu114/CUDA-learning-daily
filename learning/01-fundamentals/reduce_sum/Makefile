NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGETS := reduce_sum reduce_sum_opt

all: $(TARGETS)

reduce_sum: reduce_sum.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

reduce_sum_opt: reduce_sum_opt.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

run: all
	./reduce_sum
	./reduce_sum_opt

clean:
	rm -f $(TARGETS)
