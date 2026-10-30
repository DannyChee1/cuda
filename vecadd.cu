#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1<<24;
    size_t sz = n*sizeof(float);
    float *a = (float*)malloc(sz), *b = (float*)malloc(sz), *c = (float*)malloc(sz);
    for (int i=0; i<n; i++) { a[i] = rand()/(float)RAND_MAX; b[i] = rand()/(float)RAND_MAX; }

    float *da, *db, *dc;
    cudaMalloc(&da, sz); cudaMalloc(&db, sz); cudaMalloc(&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);

    int bs = 256, nb = (n+bs-1)/bs;
    vecAdd<<<nb, bs>>>(da, db, dc, n);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int r=0; r<20; r++) vecAdd<<<nb, bs>>>(da, db, dc, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 20;

    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i=0; i<n; i++) if (c[i] != a[i]+b[i]) bad++;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    printf("n=%d %.3fms %.1fGB/s bad=%d\n", n, ms, 3.0*sz/ms/1e6, bad);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
}
