#include "bench.cuh"
#include <vector>
#include <cublas_v2.h>

#define TILE 32


__global__ void gemm_naive(const float* A, const float* B, float* C, int N)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < N && col < N)
    {
        float sum = 0.0f;
        for(int k=0; k<N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main()
{

    const int N = 2048;

    const size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N*N);
    std::vector<float> h_B(N*N);
    std::vector<float> h_C(N*N);
    

    for(int i=0; i<N * N; i++)
    {
        h_A[i] = i % 10;
        h_B[i] = i % 7;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));
    

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    




////////////Timing and Performance for the NAIVE Kenrel///////////////////

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 threads(32 , 32);
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    for (int i=0; i<5; i++)
    {
        gemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0; i<20; i++)
    {
        gemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;

    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    total_ms = total_ms / 20;

////////////////////////////////////////////////////////////////////////




/////////////// Timing and Performance for cuBLAS///////////////////////

    cudaEvent_t start1, stop1;

    cublasHandle_t handle;
    cublasCreate(&handle);

    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));

    float alpha = 1.0f, beta = 0.0f;

    for (int i=0; i<5; i++)
    {
        cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start1));

    for (int i=0; i<20; i++)
    {
        cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop1));

    CHECK_CUDA(cudaEventSynchronize(stop1));

    float cublas_ms;

    CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start1, stop1));

    cublas_ms = cublas_ms / 20;

/////////////////////////////////////////////////////////////////////////////


    double flops = 2.0 * N * N * N;
    double gflops = flops / (total_ms/1000.0) / 1e9;

    double cublas_flops = 2.0 * N * N * N;
    double cublas_gflops = cublas_flops / (cublas_ms/1000.0) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    bool check=true;

    for(int row=0; row<20; row++)
    {
        for(int col =0; col<20; col++)
        {
            float expected = 0.0f;
            for(int k=0; k < N; k++)
            {
                expected += h_A[row * N + k] * h_B[k * N + col];
            }

            if(h_C[row * N + col] != expected) check=false;
        }
    }

    if(check) std::cout<<"Verification Passed\n";
    else std::cout<<"Verification Failed\n";

    

    std::cout << "Naive Throughput: " << gflops<<"\n" ;
    std::cout << "cuBLAS Throughput: " << cublas_gflops<<"\n" ;
    

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cublasDestroy(handle);

    return 0;


}