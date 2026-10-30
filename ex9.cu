#include <cstdio>
#include <cuda_runtime.h>

__device__ unsigned int line2, line4;

__global__ void foo_kernel(float *a, float *b, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    atomicAdd(&line2, 1);
    if (i < N) {
        b[i] = 2.7f*a[i] - 4.3f;
        atomicAdd(&line4, 1);
    }
}

int main() {
    unsigned int N = 200000;
    float *a, *b;
    cudaMalloc(&a, N*sizeof(float)); cudaMalloc(&b, N*sizeof(float));
    cudaMemset(a, 0, N*sizeof(float));

    int nb = (N + 128 - 1)/128;
    foo_kernel<<<nb, 128>>>(a, b, N);
    cudaDeviceSynchronize();

    unsigned int l2, l4;
    cudaMemcpyFromSymbol(&l2, line2, sizeof(l2));
    cudaMemcpyFromSymbol(&l4, line4, sizeof(l4));
    printf("blocks=%d threads=%d line2=%u line4=%u\n", nb, nb*128, l2, l4);

    cudaFree(a);
    cudaFree(b);
}
