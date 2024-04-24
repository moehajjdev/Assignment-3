#include <iostream>

// Define matrix dimensions
const int M = 3;  // Number of rows
const int N = 4;  // Number of columns
const int K = 5;  // Number of columns

// CUDA kernel for basic matrix multiplication
__global__ void matrixMulBasic(int* A, int* B, int* C, int M, int N, int K) {
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix multiplication
    if (row < M && col < K) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    // Allocate host memory
    int *h_A, *h_B, *h_C_basic;
    h_A = new int[M * N];
    h_B = new int[N * K];
    h_C_basic = new int[M * K];

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) h_A[i] = i + 1;
    for (int i = 0; i < N * K; ++i) h_B[i] = i + 1;

    // Allocate device memory
    int *d_A, *d_B, *d_C_basic;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * K * sizeof(int));
    cudaMalloc((void**)&d_C_basic, M * K * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridSize((K - 1) / 16 + 1, (M - 1) / 16 + 1);
    dim3 blockSize(16, 16);

    // Launch CUDA kernel for basic matrix multiplication
    matrixMulBasic<<<gridSize, blockSize>>>(d_A, d_B, d_C_basic, M, N, K);

    // Copy result from device to host
    cudaMemcpy(h_C_basic, d_C_basic, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix C (Basic):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << h_C_basic[i * K + j] << " ";
        }
        std::cout << "\n";
    }

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_basic;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_basic);

    return 0;
}
