#include "img_proc.h"

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================

void img_rgb2gray_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int sum = 0;
            for(int ch = 0; ch < channels; ch++)
            {
                sum += in[y*(width*channels) + (x*channels + ch)];
            }
            out[y*width + x] = sum/channels;
        }
    }
}

void img_invert_cpu(uchar* out, const uchar* in, const uint width, const uint height)
{
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            out[y*width + x] = 255 - in[y*width + x];
        }
    }
}

void img_blur_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int blur_size)
{
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int sum = 0;
            
            int start_x = x - blur_size/2;
            int start_y = y - blur_size/2;
            int end_x = x + blur_size/2;
            int end_y = y + blur_size/2;
            start_x = start_x >= 0 ? start_x : 0;
            start_y = start_y >= 0 ? start_y : 0;
            end_x = end_x < width ? end_x : width-1;
            end_y = end_y < height ? end_y : height-1;
            for(int kern_x = start_x; kern_x < end_x; kern_x++)
            {
                for(int kern_y = start_y; kern_y < end_y; kern_y++)
                {
                    sum += in[kern_y*width + kern_x];
                }
            }
            out[y*width + x] = sum/(blur_size*blur_size);
        }
    }
}

// =================== GPU Kernel Functions ===================
/*
TODO: Write GPU kernel functions for the above functions
   */
__global__ void img_rgb2gray_kernel(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = 0;
    if(y < height && x < width)
    {
        for(int ch = 0; ch < channels; ch++)
        {
            sum += in[y*(width*channels) + (x*channels + ch)];
        }
        out[y*width + x] = sum/channels;
    }
}

__global__ void img_invert_kernel(uchar* out, const uchar* in, const uint width, const uint height)
{
    
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(y < height && x < width)
    {
        out[y*width + x] = 255 - in[y*width + x];
    }
}

__global__ void img_blur_kernel(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size)
{
    
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = 0;
    if(y < height && x < width)
    {
        int start_x = x - blur_size/2;
        int start_y = y - blur_size/2;
        int end_x = x + blur_size/2;
        int end_y = y + blur_size/2;
        start_x = start_x >= 0 ? start_x : 0;
        start_y = start_y >= 0 ? start_y : 0;
        end_x = end_x < width ? end_x : width-1;
        end_y = end_y < height ? end_y : height-1;
        for(int kern_x = start_x; kern_x < end_x; kern_x++)
        {
            for(int kern_y = start_y; kern_y < end_y; kern_y++)
            {
                sum += in[kern_y*width + kern_x];
            }
        }
        out[y*width + x] = sum/(blur_size*blur_size);
    }
}

// =================== GPU Host Functions ===================
/* 
TODO: Write GPU host functions that launch the kernel functions above
   */
void img_rgb2gray(uchar* out, const uchar* in, const uint height, const uint width, const int channels)
{
    int block_size = 32;
    dim3 block(block_size,block_size,1);
    dim3 grid(divup(width, block_size), divup(height, block_size),1);

    img_rgb2gray_kernel<<<grid, block>>>(out, in, width, height, channels);
}

void img_invert(uchar* out, const uchar* in, const uint height, const uint width)
{
    int block_size = 32;
    dim3 block(block_size,block_size,1);
    dim3 grid(divup(width, block_size), divup(height, block_size),1);

	img_invert_kernel<<<grid, block>>>(out, in, width, height);
}

void img_blur(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size)
{
    int block_size = 32;
    dim3 block(block_size,block_size,1);
    dim3 grid(divup(width, block_size), divup(height, block_size),1);

	img_blur_kernel<<<grid, block>>>(out, in, height, width, blur_size);
}
