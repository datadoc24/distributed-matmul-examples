CC = mpicc
CFLAGS = -Wall -Wextra -std=c99 -O2
TARGET = matrix_multiply
SOURCE = matrix_multiply.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCE)
        $(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) -lm

clean:
        rm -f $(TARGET)

run: $(TARGET)
        mpirun -np 4 ./$(TARGET)

test: $(TARGET)
        @echo "Testing with 1 process:"
        mpirun -np 1 ./$(TARGET)
        @echo "\nTesting with 2 processes:"
        mpirun -np 2 ./$(TARGET)
        @echo "\nTesting with 4 processes:"
        mpirun -np 4 ./$(TARGET)

help:
        @echo "Available targets:"
        @echo "  all     - Build the matrix multiplication program"
        @echo "  clean   - Remove compiled files"
        @echo "  run     - Run the program with 4 processes"
        @echo "  test    - Run the program with different process counts"
        @echo "  help    - Show this help message"
