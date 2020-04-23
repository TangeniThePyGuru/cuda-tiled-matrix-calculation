
#include <wb.h>
#include "support.h"



#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

  __global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  const int TILE_WIDTH = 16;
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];


  // block y and x axis
  int bx = blockIdx.x; int by = blockIdx.y;
  // thread y and x axis
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // product of tile Mds[][] * Nds[][]
  float Pvalue = 0;

  for (int ph = 0; ph < ceil(numAColumns/(float)TILE_WIDTH); ++ph){

    // collaborative loading of of A and B into the shared memory with shared boundary checks
    if ((Row < numARows) && (ph*TILE_WIDTH+tx) < numAColumns)
      Mds[ty][tx] = A[Row*numAColumns + (ph*TILE_WIDTH + tx)];
    else
      Mds[ty][tx] = 0;
    if ((ph * TILE_WIDTH+ty) < numBRows && Col < numBColumns)
      Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*numBColumns + Col];
    else
      Nds[ty][tx] = 0;
    // only load the threads with a value

    // wait for all the threads to settle, before taking on the next task
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k){
        Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    // wait for all the threads to settle before moving to the next phase or completeing the loop
    __syncthreads();
  }

  // assign the pvalue to the output matrix C 
  // make sure the cell we are reading is withing bounds of matrix C
  if ((Row < numCRows) && (Col < numCColumns))
    C[Row*numCColumns + Col] = Pvalue;


}

int main(int argc, char **argv) {
  // wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  size_t A_sz, B_sz, C_sz;

  unsigned const int TILE_WIDTH = 16;

  Timer timer;
  // cudaError_t cuda_ret;

  // args = wbArg_read(argc, argv);

  // Initialize host variables ----------------------------------------------

  printf("\nInitializing host variables and creating memory on host..."); fflush(stdout);
  startTime(&timer);


  if (argc == 1) {
      numARows = 1000;
      numAColumns = numBRows = 1000;
      numBColumns = 1000;
  } else if (argc == 2) {
      numARows = atoi(argv[1]);
      numAColumns = numBRows = atoi(argv[1]);
      numBColumns = atoi(argv[1]);
  } else if (argc == 4) {
      numARows = atoi(argv[1]);
      numAColumns = numBRows = atoi(argv[2]);
      numBColumns = atoi(argv[3]);
  } else {
      printf("\n    Invalid input parameters!"
          "\n    Usage: ./lab3                # All matrices are 1000 x 1000"
          "\n    Usage: ./lab3 <m>            # All matrices are m x m"
          "\n    Usage: ./lab3 <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
          "\n");
      exit(0);
  }    

                            
  //@@ Set numCRows and numCColumns (to something other than 0, obviously)
  numCRows = numARows;
  numCColumns = numBColumns;

  // set the matrix size variables
  A_sz = numARows*numAColumns;
  B_sz = numBRows*numBColumns;
  C_sz = numARows*numBColumns;

  //@@ Allocate CPU memory and assign data

  hostA = (float*) malloc( sizeof(float)*A_sz );
  for (unsigned int i=0; i < A_sz; i++) { hostA[i] = (rand()%100)/100.00; }

  hostB = (float*) malloc( sizeof(float)*B_sz );
  for (unsigned int i=0; i < B_sz; i++) { hostB[i] = (rand()%100)/100.00; }

  hostC = (float*) malloc( sizeof(float)*C_sz );

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  printf("Allocating GPU memory..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Allocating GPU memory.");
  
  //@@ Allocate GPU memory here
  cudaMallocManaged((void **)&deviceA, sizeof(float )*A_sz);
  cudaMallocManaged((void **)&deviceB, sizeof(float )*B_sz);
  cudaMallocManaged((void **)&deviceC, sizeof(float )*C_sz);

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Allocating GPU memory.");

  printf("Copying input memory to the GPU..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float )*A_sz, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float )*B_sz, cudaMemcpyHostToDevice));
  

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns/(float)TILE_WIDTH), ceil(numCRows/(float)TILE_WIDTH), 1);
    // dimention of the block 16 X 16 block 
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1); 

  printf("Performing CUDA computation..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Compute, "Performing CUDA computation");

  printf("Copying output memory to the CPU..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Copy, "Copying output memory to the CPU");
  
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*C_sz, cudaMemcpyDeviceToHost));

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("Freeing GPU Memory.."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Freeing GPU Memory");
  
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Freeing GPU Memory");

  //Determine if output is correct and print result.
  // wbSolution(args, hostC, numCRows, numCColumns);
  verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
