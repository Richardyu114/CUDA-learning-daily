NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := naive_gemm
SRC := naive_gemm.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) 512 512 512 50

clean:
	rm -f $(TARGET)
