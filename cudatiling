#include <iostream>

// Define matrix dimensions
const int M = 3;  // Number of rows
const int N = 4;  // Number of columns
const int K = 5;  // Number of columns

#define TILE_SIZE 16

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMulTiled(int* A, int* B, int* C, int M, int N, int K) {
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tiles
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    // Initialize the accumulator for the current thread
    int sum = 0;

    // Iterate over tiles
    for (int t = 0; t < (N - 1) / TILE_SIZE + 1; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        // Synchronize threads to ensure tiles are loaded
        __syncthreads();

        // Accumulate the result for the current thread
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize threads before loading the next tiles
        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

int main() {
    // Allocate host memory
    int *h_A, *h_B, *h_C_tiled;
    h_A = new int[M * N];
    h_B = new int[N * K];
    h_C_tiled = new int[M * K];

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) h_A[i] = i + 1;
    for (int i = 0; i < N * K; ++i) h_B[i] = i + 1;

    // Allocate device memory
    int *d_A, *d_B, *d_C_tiled;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * K * sizeof(int));
    cudaMalloc((void**)&d_C_tiled, M * K * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridSize((K - 1) / TILE_SIZE + 1, (M - 1) / TILE_SIZE + 1);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);

    // Launch CUDA kernel for tiled matrix multiplication
    matrixMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, M, N, K);

    // Copy result from device to host
    cudaMemcpy(h_C_tiled, d_C_tiled, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix C (Tiled):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << h_C_tiled[i * K + j] << " ";
        }
        std::cout << "\n";
    }

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_tiled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_tiled);

    return 0;
}
