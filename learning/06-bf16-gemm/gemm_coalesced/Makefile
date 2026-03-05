NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := lesson2_gemm_coalesced
SRC := lesson2_gemm_coalesced.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) 1024 1024 1024 50

clean:
	rm -f $(TARGET)
