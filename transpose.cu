#include "bench.cuh"
#include <vector>

#define TILE 32


/*__global__ void transpose_naive(const float* in, float* out, int width, int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < height && col < width)
    {
        out[col * height + row] = in[row * width + col];
    }
}
*/

__global__ void transpose_shared(const float* in, float* out, int width, int height)
{
    __shared__ float tile[TILE][TILE+1];

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < height && col < width)
    {
        tile[threadIdx.y][threadIdx.x] = in[row * width + col];
    }
   

    __syncthreads();

    int out_col = threadIdx.x + blockIdx.y * TILE;
    int out_row = threadIdx.y + blockIdx.x * TILE;


    if(out_row < width && out_col < height)
    {
        out[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
    }


}


int main()
{
    const int WIDTH = 8192;
    const int HEIGHT = 8192;
    const int N = HEIGHT * WIDTH;

    const double PEAK_GBS = 671.0;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_in(N);
    std::vector<float> h_out(N);
    

    for(int i=0; i<N; i++)
    {
        h_in[i] = i % 1000;
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 threads(TILE , TILE);
    dim3 blocks((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);

    for (int i=0; i<5; i++)
    {
        transpose_shared<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i=0; i<100; i++)
    {
        transpose_shared<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;

    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    total_ms = total_ms / 100;

    float achieved_bw = (2 * bytes) / (total_ms / 1000) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < HEIGHT ; r += 97 )
    {
        for(int c = 0; c < WIDTH; c += 97)
        {
            if(h_out[c * HEIGHT + r] != h_in[r * WIDTH + c])
            {
                std::cout<<"Verification Failed\n";
                exit(-1);
            }
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