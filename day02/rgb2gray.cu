#include stdio.h
#include math.h

PMPP mix 3 channel pixels to get 1 gray pixel

__global__ void rgb2gray_kernel(unsigned char red, unsigned char blue, unsigned char green, unsigned char gray
int  width, int height) 
    {
    unsigned int row = blockDim.y  blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x  blockIdx.x + threadIdx.x;
    if(row  width && col  height) {
        unsigned int i = row  width + col;
        gray[i] = red[i]310 + blue[i]110 + green[i]610;
    }

}

void rgb2gray_gpu(unsigned char red, unsigned char blue, unsigned char green, unsigned char gray
    int width, int height) {
    unsigned int width, height;
    unsigned int mem = sizeof(unsigned char)  width  height

    unsigned char red_d, blue_d, green_d, gray_d;
    cudaMalloc((void)&red_d, mem);
    cudaMalloc((void)&blue_d, mem);
    cudaMalloc((void)&green_d, mem);
    cudaMalloc((void)&gray_d, mem);


    cudaMemcpy(red_d, red, mem, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, mem, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, mem, cudaMemcpyHostToDevice);


    dim3 block_size = (32, 32);
    dim3 grid_size = (width + 32 - 1  32, height + 32 - 1  32);

    rgb2graygrid_size, block_size(d_x, d_y, d_z, N);

    cudaMemcpy(gray, gray_d, mem, cudaMemcpyDeviceToHost);

    cudaFree(red_d);
    cudaFree(blue_d);
    cudaFree(green_d);
    cudaFree(gray_d);
}
