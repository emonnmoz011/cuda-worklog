#include "bench.cuh"

int main()
{
    float* p;
    CHECK_CUDA(cudaMalloc(&p, 1e18));
    std::cout<< "should not reach here\n";
    return 0;
}