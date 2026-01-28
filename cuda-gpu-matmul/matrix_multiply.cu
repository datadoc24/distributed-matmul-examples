#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int numARows, int numAColumns, int numBColumns) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < numARows && m * TILE_WIDTH + tx < numAColumns) {
            ds_A[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }

        if (Col < numBColumns && m * TILE_WIDTH + ty < numAColumns) {
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
        } else {
            ds_B[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < numARows && Col < numBColumns) {
        C[Row * numBColumns + Col] = Cvalue;
    }
}

void matrixMultiplyHost(float* A, float* B, float* C, int numARows, int numAColumns, int numBColumns) {
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numBColumns; ++j) {
            float sum = 0.0;
            for (int k = 0; k < numAColumns; ++k) {
                sum += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
            C[i * numBColumns + j] = sum;
        }
    }
}

void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)(rand() % 10);
    }
}

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <dimension of square matrix>\n", argv[0]);
        return 1;
    }


    int dimension = atoi(argv[1]);
    int numARows = dimension;
    int numAColumns = dimension;
    int numBRows = dimension;
    int numBColumns = dimension;

    size_t sizeA = numARows * numAColumns * sizeof(float);
    size_t sizeB = numBRows * numBColumns * sizeof(float);
    size_t sizeC = numARows * numBColumns * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);
    h_C_ref = (float*)malloc(sizeC);

    srand(time(NULL));
    initializeMatrix(h_A, numARows, numAColumns);
    initializeMatrix(h_B, numBRows, numBColumns);

    printf("Matrix A:\n");
    printMatrix(h_A, numARows, numAColumns);
    printf("\nMatrix B:\n");
    printMatrix(h_B, numBRows, numBColumns);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimGrid((numBColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numARows + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    clock_t cpu_start = clock();
    matrixMultiplyHost(h_A, h_B, h_C_ref, numARows, numAColumns, numBColumns);
    clock_t cpu_end = clock();
    float cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;

    printf("\nGPU Result (C = A * B):\n");
    printMatrix(h_C, numARows, numBColumns);

    printf("\nCPU Result (C = A * B):\n");
    printMatrix(h_C_ref, numARows, numBColumns);

    printf("\nMatrix dimensions: %dx%d * %dx%d = %dx%d\n", numARows, numAColumns, numBRows, numBColumns, numARows, numBColumns);
    printf("GPU time: %.2f ms\n", gpu_time);
    printf("CPU time: %.2f ms\n", cpu_time);
    
    if (cpu_time/gpu_time < 1){
            printf("\033[1;31mGPU slower\033[0m - speedup x %.2fx\n", cpu_time/gpu_time);
    }
    else {
            printf("\033[1;32mGPU faster!\033[0m - speedup x %.2fx\n", cpu_time/gpu_time);
    }

    bool correct = true;
    float tolerance = 1e-3;
    for (int i = 0; i < numARows * numBColumns && correct; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > tolerance) {
            correct = false;
        }
    }

    if (correct) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
