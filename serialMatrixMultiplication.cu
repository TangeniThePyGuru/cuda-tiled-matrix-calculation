#include <stdio.h>
#include <assert.h>
#include "support.h"

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


// void initWith(float num, float *a, int N)
// {
//   for(int i = 0; i < N; ++i)
//   {
//     a[i] = num;
//   }
// }

void multMatricesInto(float *result, float *a, float *b, int N)
{

  for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i][j] = 0.0;
			for (k = 0; k < N; k++)
				result[i][j] = c[i][j] + a[i][k] * b[k][j];
		} /* end j loop */
	}
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values multiplied correctly.\n");
}

int main(int argc, char **argv)
{
//   const int N = 2<<20;
//   size_t size = N * sizeof(float);

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
//   size_t A_sz, B_sz, C_sz;
  
  Timer timer;

  // get array sizes

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

  // set the C matrix row and col size
  numCRows = numARows;
  numCColumns = numBColumns;

  float a[numARows][numAColumns];
  float b[numARows][numBColumns];
  float c[numCRows][numCColumns];

  // matrix sizes
//   A_sz = numARows * numAColumns;
//   B_sz = numBRows * numBColumns;
//   C_sz = numCRows * numCColumns;

  //@@ Allocate CPU memory and assign data

//   a = (float*) malloc( sizeof(float)*A_sz );
  for (unsigned int i=0; j < numARows; j++) { 
    for (unsigned int j=0; i < numAColumns; i++)  {
        a[j][i] = (rand()%100)/100.00; 
    }
  }


//   b = (float*) malloc( sizeof(float)*B_sz );
  for (unsigned int j=0; j < numBRows; j++) {
    for (unsigned int i=0; i < numBColumns; i++) { 
        b[j][i] = (rand()%100)/100.00; 
    }
  }

//   c = (float*) malloc( sizeof(float)*C_sz );

  printf("Performing Mutrix Multiplication..."); fflush(stdout);
  startTime(&timer);

  multMatricesInto(c, a, b, numCRows);


  printf("Verifying Mutrix Multiplication..."); fflush(stdout);
  startTime(&timer);
  // verify
//   checkElementsAre(7, c, C_sz);

//   free(a);
//   free(b);
//   free(c);
}
    