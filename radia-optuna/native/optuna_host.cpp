// CUDA-free CPU build of the shared kernel source for hosts without CUDA.
// The same host_pdf/best code is compiled; every GPU entry point fails loudly.
#define OPTUNA_HOST_ONLY 1
#include "optuna_cuda.cu"
