## A distributed matrix multiply using MPI ##
To run it, create a hostfile in the form

myhostname1 slots=<number of CPU cores in this host>
myhostname2 slots=<number of CPU cores in this host>

...and configure passwordless SSH between all hosts.
Then:
```
mpirun -hostfile hostfile --mca btl_tcp_if_include 192.168.0.0/24 -np 10 /home/abe/mpi-tests/matrix_multiply
```
## The source code ##
Install openmpi and its dev package:

```
sudo apt install openmpi-bin libopenmpi-dev
```
Then:

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

