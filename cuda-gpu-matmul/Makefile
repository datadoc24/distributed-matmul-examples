NVCC = nvcc
CFLAGS = -O2 -std=c++11
TARGET = matrix_multiply
SOURCE = matrix_multiply.cu

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
