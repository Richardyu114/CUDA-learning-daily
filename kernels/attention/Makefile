NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := flash_attn_fwd_baseline

all: $(TARGET)

$(TARGET): flash_attn_fwd_baseline.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
