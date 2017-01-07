#include <cuda/cuda_utils.h>

cuda_timer::cuda_timer(){
    cudaEventCreate(&start_time);
}

void cuda_timer::start_timer(){
    cudaEventRecord(start_time, 0);
}

void cuda_timer::print_timer(const std::string &msg){

    cudaEvent_t curtime;
    cudaEventCreate(&curtime);
    cudaEventRecord(curtime, 0);
    cudaEventSynchronize(curtime);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_time, curtime);
    std::cout << msg << " -- Time elapsed: " << elapsed_time << "ms" << std::endl;
}

void cuda_initializer::deviceInit(int dev)
{
  int deviceCount;
  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
    exit(-1);
  }
  if (dev < 0) dev = 0;
  if (dev > deviceCount-1) dev = deviceCount - 1;

  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));
  if (deviceProp.major < 1) {
    fprintf(stderr, "cutil error: GPU device does not support CUDA.\n");
    exit(-1);
  }
  CUDA_SAFE_CALL(cudaSetDevice(dev));
}

