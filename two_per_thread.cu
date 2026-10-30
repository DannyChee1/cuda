#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void adjacent(float *a, float *b, float *c, int n) {
    int i = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    if (i < n) c[i] = a[i] + b[i];
    if (i+1 < n) c[i+1] = a[i+1] + b[i+1];
}

__global__ void sections(float *a, float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
    int j = i + blockDim.x;
    if (j < n) c[j] = a[j] + b[j];
}

typedef void (*kern)(float*, float*, float*, int);

void run(const char *name, kern k, float *da, float *db, float *dc, float *a, float *b, float *c, int n) {
    int bs = 256, nb = (n + 2*bs - 1)/(2*bs);
    size_t sz = n*sizeof(float);
    cudaMemset(dc, 0, sz);
    k<<<nb, bs>>>(da, db, dc, n);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int r=0; r<20; r++) k<<<nb, bs>>>(da, db, dc, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 20;

    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i=0; i<n; i++) if (c[i] != a[i]+b[i]) bad++;
    printf("%s %.3fms %.1fGB/s bad=%d\n", name, ms, 3.0*sz/ms/1e6, bad);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : (1<<24) + 3;
    size_t sz = n*sizeof(float);
    float *a = (float*)malloc(sz), *b = (float*)malloc(sz), *c = (float*)malloc(sz);
    for (int i=0; i<n; i++) { a[i] = rand()/(float)RAND_MAX; b[i] = rand()/(float)RAND_MAX; }

    float *da, *db, *dc;
    cudaMalloc(&da, sz); cudaMalloc(&db, sz); cudaMalloc(&dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);

    run("adjacent", adjacent, da, db, dc, a, b, c, n);
    run("sections", sections, da, db, dc, a, b, c, n);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
}
