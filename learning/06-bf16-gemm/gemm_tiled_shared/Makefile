NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := lesson3_gemm_tiled_shared
SRC := lesson3_gemm_tiled_shared.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) 1024 1024 1024 50

clean:
	rm -f $(TARGET)
