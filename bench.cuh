#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call)                                            \
                                                                    \
do{                                                                 \
    cudaError_t err__ = (call);                                     \
    if(err__ != cudaSuccess)                                        \
    {                                                               \
        std::cerr<<__FILE__<<":"<<__LINE__<<"\n";                        \
        std::cerr<<" call " << #call<<"\n";                         \
        std::cerr<<" error: "<<cudaGetErrorString(err__)<<"\n";     \
        std::exit(1);                                               \
    }                                                               \
                                                                    \
} while(0)

