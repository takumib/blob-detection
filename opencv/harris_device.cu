#include <cutil.h>
#include <stdio.h>
#include <stdlib.h>

#define THRESH 70000.0
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define KERNEL_RADIUS 1
#define BLOCKSIZE 5

typedef char guchar;

__constant__ int gx[3][3] = { {-1, 0, 1},
							  {-2, 0, 2},
							  {-1, 0, 1}, };

__constant__ int gy[3][3] = { {-1, -2, -1},
							  { 0,  0,  0},
							  { 1,  2,  1}, };

__device__ void gpu_compute_xgrad(guchar *device_image, int width, int height, guchar *device_x_component);

 
__global__ void gpu_average_blur(float *device_image, int width, int height) {

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int r, c;

	int sum = 0;
	int n = 0;

	for(r = MAX(0, iy-1); r <= MIN(height-1,iy+1); r++)
	{
		for(c = MAX(0, ix-1); c <= MIN(width-1,ix+1); c++)
		{
			sum += device_image[r * width + c];
			n++;
		}
	}
		
	device_image[iy * width + ix] = sum / n;
}

__global__ void gpu_compute_xgrad(guchar *device_image, int width, int height, guchar *device_x_component)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;
	int i, j, r, c;

	for(r = MAX(0, iy-1), i = 0; r <= MIN(height-1,iy+1); r++, i++)
	{
		for(c = MAX(0, ix-1), j = 0; c <= MIN(width-1,ix+1); c++, j++)
		{
			sum += device_image[r * width + c] * gx[i][j];
		}
	}
	
	device_x_component[iy * width + ix] = sum;
}

__global__ void gpu_compute_ygrad(guchar *device_image, int width, int height, guchar *device_y_component)
{

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;
	int i, j, r, c;

	for(r = MAX(0, iy-1), i = 0; r <= MIN(height-1,iy+1); r++, i++)
	{
		for(c = MAX(0, ix-1), j = 0; c <= MIN(width-1,ix+1); c++, j++)
		{
			sum += device_image[r * width + c] * gy[i][j];
		}
	}
	
	device_y_component[iy * width + ix] = sum;
}

__global__ void gpu_gradient_matrix(guchar *x_component, guchar *y_component, int width, int height, float *xx_grad, float *xy_grad, float *yy_grad)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	xx_grad[iy * width + ix] = x_component[iy * width + ix] * x_component[iy * width + ix];
	xy_grad[iy * width + ix] = x_component[iy * width + ix] * y_component[iy * width + ix];
	yy_grad[iy * width + ix] = y_component[iy * width + ix] * y_component[iy * width + ix];
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


__global__ void gpu_threshold(guchar *device_image, int width, int height, float *harris)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if(harris[iy * width + ix] > THRESH)
	{
		device_image[iy * width + ix] = (guchar)255;
	}
}
