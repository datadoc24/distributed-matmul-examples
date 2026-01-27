 1. matrix_multiply.c - A C program that:
    - Creates two 100x100 matrices with random floating point values between 0 and 1
    - Uses MPI to distribute matrix multiplication across multiple processes
    - Each process prints debug messages showing its rank and progress
    - Process 0 initializes matrices and gathers final results
    - All processes participate in the distributed computation
  2. Makefile - With targets for:
    - make or make all - Compile the program
    - make run - Run with 4 processes
    - make test - Test with 1, 2, and 4 processes
    - make clean - Remove compiled files

  The program compiled successfully. You can now run it with:
  mpirun -np 4 ./matrix_multiply

  Or use the Makefile targets:
  make run    # Run with 4 processes
  make test   # Test with different process counts
