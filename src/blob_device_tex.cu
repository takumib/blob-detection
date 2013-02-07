#include "blob_device.hpp"

__global__ void tex_gpu_gaussian_blur(int width, int height, float sigma, float *weights, float *src, float *dest) {

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int i, j;
	int row, col;

	float sum = 0;

	int radius = (int)floor((3 * sigma) + 0.5);//int_round(3 * sigma);

	int kernelSize = 2 * radius + 1;

	sum = 0;

	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			row = abs(iy+i);
			col = abs(ix+j);

			float c = tex1Dfetch(tex_img, row * width + col);

			sum += weights[(i+radius) * kernelSize + (j+radius)] * c;//src[row * width + col];
		}
	}	

	dest[iy * width + ix] = sum;

}

__global__ void tex_gpu_gaussian_diff(int width, int height, float *g1, float *g2, float *dest)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	dest[iy * width + ix] = g2[iy * width + ix] - g1[iy * width + ix];
}

