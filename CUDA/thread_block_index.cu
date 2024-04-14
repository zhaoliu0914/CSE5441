#include<stdio.h>
#include<cuda.h>

__global__ void myKernelMethod(void) {
    printf("Start running myKernelMethod method().\n");
    printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d,  blockDim.x = %d, blockDim.y = %d, blockDim.z = %d \n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d \n",
           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

    printf("Finish running myKernelMethod method().\n");
}

int main(void) {
    printf("Start running main method().\n");

    dim3 dimBlock(2, 3);
    dim3 dimGrid(4, 2);

    myKernelMethod<<<dimGrid, dimBlock>>>();

    printf("Finish running main method().\n");

    cudaDeviceSynchronize();

    return 0;
}