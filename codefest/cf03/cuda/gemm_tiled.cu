// Tiled GEMM kernel: shared-memory tiling with tile size T=8
// Matrix size: 1024x1024 FP32, row-major storage
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define T 8

__global__ void gemm_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + T - 1) / T; t++) {
        int a_col = t * T + threadIdx.x;
        int b_row = t * T + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < T; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 17) / 17.0f;
        h_B[i] = (float)(i % 13) / 13.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (N + T - 1) / T);

    // Warmup
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 10;
    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) {
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= runs;

    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms * 1e-3) / 1e9;

    printf("Tiled GEMM (%dx%d FP32, tile=%d)\n", N, N, T);
    printf("  Avg time: %.3f ms\n", ms);
    printf("  Throughput: %.2f GFLOP/s\n", gflops);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
