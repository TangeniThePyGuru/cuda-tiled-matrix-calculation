
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
  wbArg_t args;
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
  
  unsigned const int TILE_WIDTH = 16;

  Timer timer;
  // cudaError_t cuda_ret;

  args = wbArg_read(argc, argv);

  // Initialize host variables ----------------------------------------------

  printf("\nImporting data and creating memory on host..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
                            
  //@@ Set numCRows and numCColumns (to something other than 0, obviously)
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*) malloc( sizeof(float)*(numCColumns*numCRows) );

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  printf("Allocating GPU memory..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Allocating GPU memory.");
  
  //@@ Allocate GPU memory here
  cudaMallocManaged((void **)&deviceA, sizeof(float )*(numAColumns*numARows));
  cudaMallocManaged((void **)&deviceB, sizeof(float )*(numBColumns*numBRows));
  cudaMallocManaged((void **)&deviceC, sizeof(float )*(numCColumns*numCRows));

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Allocating GPU memory.");

  printf("Copying input memory to the GPU..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float )*(numAColumns*numARows), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float )*(numBColumns*numBRows), cudaMemcpyHostToDevice));
  

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
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*(numCRows*numCColumns), cudaMemcpyDeviceToHost));

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
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
