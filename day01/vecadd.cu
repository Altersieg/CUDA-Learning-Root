#include <stdio.h>
#include <math.h>

__global__ void vecadd(double* d_x, double* d_y, double* d_z, int N) {
    for(int i = 0; i < N; ++i) {
        d_z[i] = d_x[i] + d_y[i];
    }
}

int main() {
    int N = 10000;
    int mem = sizeof(double) * N;

    double* h_x = (double*)malloc(mem);
    double* h_y = (double*)malloc(mem);
    double* h_z = (double*)malloc(mem);

    for(int i = 0; i<N; ++i) {
        h_x[i] = rand();
        h_y[i] = rand();
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, mem);
    cudaMalloc((void**)&d_y, mem);
    cudaMalloc((void**)&d_z, mem);
    cudaMemcpy(d_x, h_x, mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, mem, cudaMemcpyHostToDevice);

    int block_size = 512;
    int grid_size = (N + block_size -1) / block_size;
    vecadd<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    cudaMemcpy(h_z, d_z, mem, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
