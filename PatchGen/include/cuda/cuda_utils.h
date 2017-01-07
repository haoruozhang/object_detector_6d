/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Cuda helper functions
 */
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(errorMessage)                                  \
  do									\
  {                                                                     \
    cudaError_t err = cudaGetLastError();                               \
    if(err != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr,"Cuda error: %s in file '%s' in line %i : %s.\n",  \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err)); \
      fprintf(stderr,"Press ENTER key to terminate the program\n");    \
      getchar();                                                        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
    err = cudaThreadSynchronize();                                      \
    if(err != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr,"Cuda error: %s in file '%s' in line %i : %s.\n",  \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err)); \
      fprintf(stderr,"Press ENTER key to terminate the program\n");     \
      getchar();                                                        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }                                                                     \
  while(0)

// Call and check function for CUDA error without synchronization
#define CUDA_SAFE_CALL_NO_SYNC(call)                                    \
  do                                                                    \
  {                                                                     \
    cudaError err = call;                                               \
    if(err != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      fprintf(stderr,"Press ENTER key to terminate the program\n");     \
      getchar();                                                        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }                                                                     \
  while(0)

// Call and check function for CUDA error with synchronization
#define CUDA_SAFE_CALL(call)                                            \
  do                                                                    \
  {                                                                     \
    CUDA_SAFE_CALL_NO_SYNC(call);                                       \
    cudaError err = cudaThreadSynchronize();                            \
    if(err != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr,"Cuda errorSync in file '%s' in line %i : %s.\n",  \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      fprintf(stderr,"Press ENTER key to terminate the program\n");     \
      getchar();                                                        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define CUDA_SAFE_CALL(call)                                            \
  do                                                                    \
  {                                                                     \
    CUDA_SAFE_CALL_NO_SYNC(call);                                       \
    cudaError err = cudaThreadSynchronize();                            \
    if(err != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr,"Cuda errorSync in file '%s' in line %i : %s.\n",  \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      fprintf(stderr,"Press ENTER key to terminate the program\n");     \
      getchar();                                                        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)




class cuda_timer{

    cudaEvent_t start_time;

public:
    cuda_timer();
    void start_timer();
    void print_timer(const std::string &msg);

};

class cuda_initializer{

public:
    cudaDeviceProp deviceProp;

    void deviceInit(int dev = 0);
};


#endif
