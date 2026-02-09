NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := saxpy

all: $(TARGET)

$(TARGET): saxpy.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
