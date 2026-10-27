#include "bench.cuh"
#include <vector>

#define BLOCK 256
#define RADIUS 3


static const int threads = 256;

/*
__global__ void stencil_naive(const float* in, float* out, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i<n; i+=stride)
    {
        float sum = 0.0f;
        if(i >= RADIUS && i < n - RADIUS)
        {
            for (int off = -RADIUS; off <= RADIUS; off++)
            {
                sum = sum + in[i + off];
            }

            out[i] = sum;
        }
    }
}
*/


__global__ void stencil_shared(const float* in, float* out, int n)
{
   __shared__ float tile[BLOCK + 2 * RADIUS];

   int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
   int local_idx = threadIdx.x + RADIUS;

   tile[local_idx] = in[global_idx];

   if (threadIdx.x < RADIUS)
   {
        if(global_idx >= RADIUS)
        {
        tile[threadIdx.x] = in[global_idx - RADIUS];
        }
        else
        {
            tile[threadIdx.x] = 0.0f;
        }


        if(global_idx + BLOCK < n) 
        {
        tile[local_idx + BLOCK] = in [global_idx + BLOCK];
        }
        else 
        {
            tile[local_idx + BLOCK] = 0.0f;
        }
   }

   __syncthreads();

    float sum = 0.0f;

   for (int off = -RADIUS;  off <= RADIUS; off++)
   {
       
        sum = sum + tile [local_idx + off];
   }

   out[global_idx] = sum;
}


int main()
{
    const int N = 100000000;
    const double PEAK_GBS = 671.0;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_in(N);
    std::vector<float> h_out(N);
    

    for(int i=0; i<N; i++)
    {
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i=0; i<5; i++)
    {
        stencil_shared<<<blocks, threads>>>(d_in, d_out, N);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0; i<100; i++)
    {
        stencil_shared<<<blocks, threads>>>(d_in, d_out, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;

    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    total_ms = total_ms / 100;

    float achieved_bw = (2 * bytes) / (total_ms / 1000) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = RADIUS; i < N - RADIUS; i++)
    {
        float expected = 0.0f;
        for (int off = -RADIUS; off <= RADIUS; off++)
        {
            expected = expected + h_in[i + off];
        }

        if (h_out[i] != expected)
        {
            std::cout << "Verification Failed\n";
            exit(-1);
        }
    }

    std::cout<<"Verification Passed\n";

    std::cout << "Bandwidth: " << achieved_bw << " GB/s\n";
    std::cout << "Percent of peak: " << (achieved_bw / PEAK_GBS) * 100 << "%\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;


}