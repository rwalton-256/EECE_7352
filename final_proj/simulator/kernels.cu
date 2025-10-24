#include <stdint.h>

__global__ void vecmul( float* buf1, float* buf2 )
{
	auto idx = threadIdx.x + blockDim.x * blockIdx.x;
	buf1[idx] = buf1[idx] * buf2[idx];
}

constexpr uint32_t mat_dim = 64;

__global__ void mattranspose( float* buf1, float* buf2 )
{
	auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    auto idx_i = idx / mat_dim;
    auto idx_j = idx % mat_dim;
	buf2[ idx_j * mat_dim + idx_i ] = buf1[ idx_i * mat_dim + idx_j ];
}

__global__ void matmul( float* buf1, float* buf2, float* buf3 )
{
	auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    auto idx_i = idx / mat_dim;
    auto idx_j = idx % mat_dim;
    float sum = 0;
    for( size_t k=0; k<mat_dim; k++ )
    {
        sum += buf2[ k * mat_dim + idx_j ] * buf1[ idx_i * mat_dim + k ];
    }
    buf3[ idx_i * mat_dim + idx_j ] = sum;
}


int main()
{}
