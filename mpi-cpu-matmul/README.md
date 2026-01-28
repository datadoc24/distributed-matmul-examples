## A distributed matrix multiply using MPI ##
A C program that:
- Creates two 1000x1000 matrices with random floating point values between 0 and 1
- Uses MPI to distribute matrix multiplication across multiple processes
- Each process prints debug messages showing its rank and progress
- Process 0 initializes matrices and gathers final results
- All processes participate in the distributed computation

### Install MPI and build the program ###
Install openmpi and its dev package:

```
cd mpi-cpu-matmul
sudo apt install -y openmpi-bin libopenmpi-dev
make all
```
### Set up distributed hosts and mpirun the program ###
    
To run it, create a hostfile in the form
```
myhostname1 slots=12
myhostname2 slots=12
```
..where the number of slots equals the number of cores on the host that you want allow MPI processes to be allocated to.

Configure passwordless SSH between all hosts

SCP the compiled binary to an identically-named location on all the hosts

Use mpirun to execute the program

```
mpirun -hostfile hostfile --mca btl_tcp_if_include 192.168.0.0/24 -np 10 /home/abe/mpi-tests/matrix_multiply
```
Explanation of parameters:
hostfile: the hostfile containing the resolvable DNS hostnames (or IP addresses) of the compute hosts that will run the distributed processes;
btl_tcp_if_include: a filter that determines what network interfaces on each host will be used for inter-process SSH comms. Because the IP interface names might differ on each host, I found that using the subnet works best. 'btl' stands for 'byte transport layer' in MPI-speak.
np: Number of processes to split the workload into. In this matrix example, the matrix 'SIZE' parameter must be divisible by the number of processes.
Finally: give the path to the executable, in a way that works for each of the hosts names in *hostfile*
