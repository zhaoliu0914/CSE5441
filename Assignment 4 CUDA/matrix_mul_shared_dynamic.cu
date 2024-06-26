/* Copyright (c) 1993-2015, CS Department of OSU. All rights reserved.*/
#include <stdio.h>
// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


/* You should not change the value of DSIZE */
const int DSIZE = 18432;
int block_size = 8;
#define TILE_WIDTH 4
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {
  __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH]; // define static shared memory in CUDA
  __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH]; // define static shared memory in CUDA
  int idx = threadIdx.x + blockDim.x * blockIdx.x;  // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y;  // create thread y index

  if ((idx < ds) && (idy < ds)) {
    float temp = 0;
    for (int i = 0; i < ds / TILE_WIDTH; ++i) {
        sharedA[threadIdx.y][threadIdx.x] = A[idy * ds + (i * TILE_WIDTH + threadIdx.x)]; // copy value from global memory to shared memory
        sharedB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * ds + idx]; // copy value from global memory to shared memory
        __syncthreads();  // Synchronizes all threads within a block

        for (int k = 0; k < TILE_WIDTH; ++k) {
            temp += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x]; // perform inner product within tile/submatrix
        }
        __syncthreads();  // Synchronizes all threads within a block
    }
    C[idy * ds + idx] = temp;
  }
}

int main(int argc, char *argv[]) {
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;

  if (argc == 2) {
      block_size = atoi(argv[1]);
      if (block_size <= 0) {
          fprintf(stderr, "Error: block_size should be >= 1\n");
          exit(1);
      }
  }

  // start timing
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];
  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = static_cast<double>(t1 - t0) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(TILE_WIDTH, TILE_WIDTH);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H failure");

  // GPU timing
  t2 = clock();
  t2sum = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  for (int i = 0; i < DSIZE * DSIZE; i++) {
      if (h_C[i] != A_val * B_val * DSIZE) {
          printf("mismatch at index %d, was: %f, should be: %f\n",
                  i, h_C[i], A_val * B_val * DSIZE);
          return -1;
      }
  }
  printf("Success!\n");
  return 0;
}

