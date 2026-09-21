#include "bench.cuh"
#include <vector>

static const int threads = 256;

static __global__ void vadd(const float* const a, const float* const b, float* const c, int n)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int stride = gridDim.x * blockDim.x;

    for(int i=idx; i<n; i += stride)

    if(i<n) c[i] = a[i] + b[i];
}

int main()
{
    const int N = 100000000;
    const double PEAK_GBS = 224.0;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for(int i=0; i<N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 0.5f;
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i=0; i<5; i++)
    {
        vadd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0; i<100; i++)
    {
        vadd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;

    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    total_ms = total_ms / 100;

    float achieved_bw = (3 * bytes) / (total_ms / 1000) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    for(int i=0; i<N; i++)
    {
        if(h_c[i] != h_a[i] + h_b[i]) 
        {
            std::cout<<"Verification Failed";
            exit(-1);
        }
    }

    std::cout<<"Verification Passed\n";

    std::cout << "Bandwidth: " << achieved_bw << " GB/s\n";
    std::cout << "Percent of peak: " << (achieved_bw / PEAK_GBS) * 100 << "%\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;


}