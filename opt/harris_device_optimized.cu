#include <cutil.h>
#include <stdio.h>
#include <stdlib.h>

#define THRESH 75000.0
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define BLOCKSIZE 8

__global__ void gpu_average_blur(unsigned char *device_image, int width, int height) {

	const int bx = blockIdx.x; const int by = blockIdx.y;
	const int tx = threadIdx.x; const int ty = threadIdx.y;

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	
	int sum = 0;
	int n = 0;

	int r, c;
	
	__shared__ unsigned char shared_device_image[BLOCKSIZE * BLOCKSIZE];

	shared_device_image[ty * width + tx] = device_image[iy * width + ix];

	__syncthreads();
	
	for(r = MAX(0, ty-1); r <= MIN(BLOCKSIZE-1,ty+1); r++)
	{
		for(c = MAX(0, tx-1); c <= MIN(BLOCKSIZE-1,tx+1); c++)
		{
			sum += shared_device_image[r * width + c];
			n++;
		}
	}

	__syncthreads();
		
	device_image[iy * width + ix] = sum / n;
}

__global__ void gpu_compute_xgrad(unsigned char *device_image, int width, int height, unsigned char *device_x_component)
{
	const int bx = blockIdx.x; const int by = blockIdx.y;
	const int tx = threadIdx.x; const int ty = threadIdx.y;

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;
	int i, j, r, c;

	int gx[3][3] = { {-1, 0, 1},
					 {-2, 0, 2},
					 {-1, 0, 1} };

	__shared__ unsigned char shared_device_image[BLOCKSIZE * BLOCKSIZE];
	shared_device_image[ty * width + tx] = device_image[iy * width + ix];

	__syncthreads();

	for(r = MAX(0, ty-1); r <= MIN(BLOCKSIZE-1,ty+1); r++)
	{
		for(c = MAX(0, tx-1); c <= MIN(BLOCKSIZE-1,tx+1); c++)
		{
			sum += shared_device_image[r * width + c] * gx[i][j];
		}
	}
	
	__syncthreads();

	device_x_component[iy * width + ix] = sum;

}


__global__ void gpu_compute_ygrad(unsigned char *device_image, int width, int height, unsigned char *device_y_component)
{
	const int bx = blockIdx.x; const int by = blockIdx.y;
	const int tx = threadIdx.x; const int ty = threadIdx.y;

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;
	int i, j, r, c;

	int gy[3][3] = { {-1, -2, -1},
					 {0, 0, 0},
					 {1, 2, 1} };

	__shared__ unsigned char shared_device_image[BLOCKSIZE * BLOCKSIZE];
	shared_device_image[ty * width + tx] = device_image[iy * width + ix];

	__syncthreads();

	for(r = MAX(0, ty-1), i = 0; r < MIN(BLOCKSIZE-1,ty+1); r++, i++)
	{
		for(c = MAX(0, tx-1), j = 0; c < MIN(BLOCKSIZE-1,tx+1); c++, j++)
		{
			sum += shared_device_image[r * width + c] * gy[i][j];
		}
	}

	__syncthreads();
	
	device_y_component[iy * width + ix] = sum;

}


__global__ void gpu_gradient_matrix(int width, int height, unsigned char *x_component, unsigned char *y_component, float *xx_grad, float *xy_grad, float *yy_grad)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int r, c;
	
	float xx_sum = 0.0;
	float xy_sum = 0.0;
	float yy_sum = 0.0;

	for(r = MAX(0, ty-1); r <= MIN(BLOCKSIZE-1,ty+1); r++)
	{
		for(c = MAX(0, tx-1); c <= MIN(BLOCKSIZE-1,tx+1); c++)
		{
			xx_sum += x_component[r * width + c] * x_component[r * width + c];
			xy_sum += x_component[r * width + c] * y_component[r * width + c];
			yy_sum += y_component[r * width + c] * y_component[r * width + c];
		}
	}

	
	xx_grad[iy * width + ix] = xx_sum;
	xy_grad[iy * width + ix] = xy_sum;
	yy_grad[iy * width + ix] = yy_sum;
}


__global__ void gpu_harris_corner(int width, int height, float *harris, float *xx_grad, float *xy_grad, float *yy_grad)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	
	float det = xx_grad[iy * width + ix] * yy_grad[iy * width + ix] - (xy_grad[iy * width + ix] * xy_grad[iy * width + ix]);
	float trace = xx_grad[iy * width + ix] + yy_grad[iy * width + ix];

	harris[iy * width + ix] = det - 0.06 * (trace * trace);
}

__global__ void gpu_nonmax_suppression(int width, int height, float *harris)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int r, c;
	
	float curr = harris[iy * width + ix];

	if(harris[iy * width + ix] < 0.0)
	{
		harris[iy * width + ix] = 0.0;
	}
   
	for(r = MAX(0, iy-1); r <= MIN(height-1,iy+1); r++)
	{
		for(c = MAX(0, ix-1); c <= MIN(width-1,ix+1); c++)
		{
			if(curr > harris[r * width + c])
			{
				harris[r * width + c] = 0;//curr - harris[r * width + c];
			}	
			else
			{	
				curr = harris[r * width + c];
			}
		}
	}
}


__global__ void gpu_threshold(unsigned char *device_image, int width, int height, float *harris)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	//device_image[iy * width + ix] = 0;
	if(harris[iy * width + ix] > THRESH)// && harris[iy * width + ix] < 1000000000.0)
	{
		device_image[iy * width + ix] = 255;
	}
}
