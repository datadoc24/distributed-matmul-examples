#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SIZE 1000

void initialize_matrix(float *matrix) {
    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = (float)rand() / RAND_MAX;
        }
    }
}

void print_matrix_sample(float *matrix, int rank) {
    printf("Process %d - Matrix sample (first 5x5):\n", rank);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.3f ", matrix[i * SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    float *A, *B, *C;
    float *local_A, *local_C;
    int rows_per_process;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Process %d of %d started\n", rank, size);

    rows_per_process = SIZE / size;
    if (SIZE % size != 0) {
        if (rank == 0) {
            printf("ERROR: Matrix size %d must be divisible by number of processes %d\n", SIZE, size);
        }
        MPI_Finalize();
        return 1;
    }

    A = malloc(SIZE * SIZE * sizeof(float));
    B = malloc(SIZE * SIZE * sizeof(float));
    C = malloc(SIZE * SIZE * sizeof(float));
    local_A = malloc(rows_per_process * SIZE * sizeof(float));
    local_C = malloc(rows_per_process * SIZE * sizeof(float));

    if (!A || !B || !C || !local_A || !local_C) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Process %d: Initializing matrices...\n", rank);
        initialize_matrix(A);
        initialize_matrix(B);
        printf("Process %d: Matrices initialized\n", rank);
        print_matrix_sample(A, rank);
    }

    printf("Process %d: Broadcasting matrix B...\n", rank);
    MPI_Bcast(B, SIZE * SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("Process %d: Matrix B received\n", rank);

    printf("Process %d: Scattering matrix A rows...\n", rank);
    MPI_Scatter(A, rows_per_process * SIZE, MPI_FLOAT,
                local_A, rows_per_process * SIZE, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    printf("Process %d: Received %d rows of matrix A\n", rank, rows_per_process);

    printf("Process %d: Starting matrix multiplication...\n", rank);
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < SIZE; j++) {
            local_C[i * SIZE + j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                local_C[i * SIZE + j] += local_A[i * SIZE + k] * B[k * SIZE + j];
            }
        }
    }
    printf("Process %d: Matrix multiplication completed\n", rank);

    printf("Process %d: Gathering results...\n", rank);
    MPI_Gather(local_C, rows_per_process * SIZE, MPI_FLOAT,
               C, rows_per_process * SIZE, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process %d: All results gathered\n", rank);
        print_matrix_sample(C, rank);
        printf("Process %d: Matrix multiplication complete!\n", rank);

        printf("Process %d: Result verification - C[0][0] = %.6f\n", rank, C[0]);
        if (SIZE > 50) {
            printf("Process %d: Result verification - C[50][50] = %.6f\n", rank, C[50 * SIZE + 50]);
        }
        if (SIZE > 99) {
            printf("Process %d: Result verification - C[99][99] = %.6f\n", rank, C[99 * SIZE + 99]);
        }
    }

    printf("Process %d: Cleaning up and finalizing\n", rank);
    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    printf("Process %d: Finished\n", rank);
    return 0;
}
