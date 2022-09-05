#include <stdio.h>
#include <stdlib.h>

#include "matrixmul.h"
#include "timer.h"

#define BLOCK_SIZE 16

__global__ void block_mm_kernel(const float* A, const float* B, float* output, int M, int N) 
{
	// TODO: complete the block matrix kernel function
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE];
    
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    int blocks = M / BLOCK_SIZE;
    
    float sum = 0;
    for(int block = 0; block < blocks; block++)
    {
        blockA[threadIdx.y][threadIdx.x] = A[(y * M) + (BLOCK_SIZE*block + threadIdx.x)];
        blockB[threadIdx.y][threadIdx.x] = B[((BLOCK_SIZE*block + threadIdx.y) * N) + (x)];
        
        __syncthreads();
        
        for(int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += blockA[threadIdx.y][i] * blockB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    output[(y * N) + (x)] = sum;
}


inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}


float run_mm_gpu(const float* A, const float* B, float* C, int M, int N)
{
    // sanity checks
    if(M % BLOCK_SIZE != 0)
    {
        printf("M value of %d is not a multiple of block size %d!\n", M, BLOCK_SIZE);
        return 0;
    }
    
    if(N % BLOCK_SIZE != 0)
    {
        printf("N value of %d is not a multiple of block size %d!\n", N, BLOCK_SIZE);
        return 0;
    }
    
	Timer gpu_timer;
	gpu_timer.start();

	//TODO: launch the kernel function
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(divup(N, BLOCK_SIZE), divup(N, BLOCK_SIZE), 1);
    block_mm_kernel<<<grid, block>>>(A, B, C, M, N);
	
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}


