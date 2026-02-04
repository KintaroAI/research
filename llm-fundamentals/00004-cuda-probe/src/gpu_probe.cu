/*
 * GPU Probe - Simple CUDA test to verify GPU is working
 * Experiment 00004
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel: vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Kernel to square each element (tests computation)
__global__ void vectorSquare(float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * a[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    printf("=== GPU Probe - CUDA Test ===\n\n");

    // 1. Check CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA Devices found: %d\n\n", deviceCount);

    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }

    // 2. Print device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  SM count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Memory clock: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("\n");
    }

    // 3. Test memory allocation and kernel execution
    printf("--- Running computation test ---\n");
    
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");
    printf("GPU memory allocated: %.2f MB x 3\n", bytes / (1024.0 * 1024.0));
    
    // Copy to device
    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "memcpy H2D a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "memcpy H2D b");
    printf("Data copied to GPU\n");
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    
    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "device sync");
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel executed in %.4f ms\n", milliseconds);
    
    // Copy result back
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "memcpy D2H");
    
    // Verify result
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            errors++;
            if (errors < 10) {
                printf("ERROR at %d: expected 3.0, got %.2f\n", i, h_c[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("✓ Verification PASSED: all %d elements correct!\n", N);
    } else {
        printf("✗ Verification FAILED: %d errors\n", errors);
    }
    
    // 4. Benchmark: run many iterations
    printf("\n--- Benchmark: 1000 kernel launches ---\n");
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("1000 kernels in %.2f ms (%.2f us per kernel)\n", milliseconds, milliseconds * 1000 / 1000);
    
    // Compute bandwidth
    float gigaOps = (float)N * 1000 / 1e9;  // 1M elements * 1000 iterations
    printf("Throughput: %.2f billion elements/sec\n", gigaOps / (milliseconds / 1000));
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\n=== GPU Probe Complete ===\n");
    printf("✓ CUDA is working!\n");
    
    return 0;
}
