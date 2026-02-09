NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
TARGET := vector_add

all: $(TARGET)

$(TARGET): vector_add.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
