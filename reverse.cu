#include "bench.cuh"
#include <vector>

static const int threads = 256;

static __global__ void reverse(const float* const in, float* const out, int n)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int stride = gridDim.x * blockDim.x;

    for(int i=idx; i<n; i += stride)

    if(i<n) out[n-i-1] = in[i];
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
        h_in[i] = i;
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
        reverse<<<blocks, threads>>>(d_in, d_out, N);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0; i<100; i++)
    {
        reverse<<<blocks, threads>>>(d_in, d_out, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;

    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    total_ms = total_ms / 100;

    float achieved_bw = (2 * bytes) / (total_ms / 1000) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    for(int i=0; i<N; i++)
    {
        if(h_out[i] != h_in[N -i -1]) 
        {
            std::cout<<"Verification Failed";
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