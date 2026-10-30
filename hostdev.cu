#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ float f(float x) {
    return 2.7f*x - 4.3f;
}

__global__ void apply(float *a, float *b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) b[i] = f(a[i]);
}

int main() {
    int n = 1000;
    size_t sz = n*sizeof(float);
    float *a = (float*)malloc(sz), *b = (float*)malloc(sz);
    for (int i=0; i<n; i++) a[i] = rand()/(float)RAND_MAX;

    float *da, *db;
    cudaMalloc(&da, sz); cudaMalloc(&db, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    apply<<<(n+255)/256, 256>>>(da, db, n);
    cudaMemcpy(b, db, sz, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i=0; i<n; i++) if (fabsf(b[i] - f(a[i])) > 1e-5f) bad++;
    printf("bad=%d\n", bad);

    cudaFree(da);
    cudaFree(db);
    free(a);
    free(b);
}
