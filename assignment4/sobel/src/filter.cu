#include "filter.h"
#include "timer.h"

#include <iostream>

using namespace std;

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width)
{
    const int sobel_kernel_x[3][3] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1}};

   const int sobel_kernel_y[3][3] = {
        { 1,    2,  1},
        { 0,    0,  0},
        { -1,  -2, -1}};
        
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int sum_x = 0;
            int sum_y = 0;
            int start_x = x > 0 ? 0 : 1;
            int start_y = y > 0 ? 0 : 1;
            int end_x = x < width-1 ? 3 : 2;
            int end_y = y < height-1 ? 3 : 2;
            for(int kern_x = start_x; kern_x < end_x; kern_x++)
            {
                for(int kern_y = start_y; kern_y < end_y; kern_y++)
                {
                    sum_x += sobel_kernel_x[kern_x][kern_y]*input[(y+kern_y-1)*width + (x+kern_x-1)];
                    sum_y += sobel_kernel_y[kern_x][kern_y]*input[(y+kern_y-1)*width + (x+kern_x-1)];
                }
            }
            unsigned int total = sqrt(sum_x*sum_x + sum_y*sum_y);
            output[y*width + x] = total > 255 ? 255 : total;
            //output[y*width + x] = 255;
            //output[y*width + x] = input[y*width + x];
        }
    }
}

// =================== GPU Kernel Functions ===================
__global__ void img_sobel_kernel(uchar* out, const uchar* in, const uint height, const uint width)
{
    __const__ int sobel_kernel_x[3][3] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1}};

    __const__ int sobel_kernel_y[3][3] = {
        { 1,    2,  1},
        { 0,    0,  0},
        { -1,  -2, -1}};
    
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sum_x = 0;
    int sum_y = 0;
    if(y < height && x < width)
    {
        int start_x = x > 0 ? 0 : 1;
        int start_y = y > 0 ? 0 : 1;
        int end_x = x < width-1 ? 3 : 2;
        int end_y = y < height-1 ? 3 : 2;
        for(int kern_x = start_x; kern_x < end_x; kern_x++)
        {
            for(int kern_y = start_y; kern_y < end_y; kern_y++)
            {
                sum_x += sobel_kernel_x[kern_x][kern_y]*in[(y+kern_y-1)*width + (x+kern_x-1)];
                sum_y += sobel_kernel_y[kern_x][kern_y]*in[(y+kern_y-1)*width + (x+kern_x-1)];
            }
        }
        unsigned int total = sqrtf((float)(sum_x*sum_x + sum_y*sum_y));
        out[y*width + x] = total > 255 ? 255 : total;
    }
}

// =================== GPU Host Functions ===================
void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	int block_size = 32;
    dim3 block(block_size,block_size,1);
    dim3 grid(divup(width, block_size), divup(height, block_size),1);

	img_sobel_kernel<<<grid, block>>>(output, input, height, width);
    
    int result = cudaDeviceSynchronize();
    if(result != 0)
    {
        cout << "Sobel kernel failed with result: " << result << endl;
    }
}
