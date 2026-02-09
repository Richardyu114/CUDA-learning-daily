NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := reduce_sum

all: $(TARGET)

$(TARGET): reduce_sum.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
