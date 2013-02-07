#include <cutil.h>
#include <stdio.h>
#include <stdlib.h>

#define THRESH 70000.0
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define RADIUS 1
#define BLOCKSIZE 16

typedef char guchar;

__constant__ int gx[3][3] = { {-1, 0, 1},
							  {-2, 0, 2},
							  {-1, 0, 1}, };

__constant__ int gy[3][3] = { {-1, -2, -1},
							  { 0,  0,  0},
							  { 1,  2,  1}, };
 

__shared__ guchar temp[BLOCKSIZE + 2 * RADIUS + 2][BLOCKSIZE + 2 * RADIUS + 2];

__device__ void gpu_sobel_op(int y, int x, int iy, int ix, int width, guchar *device_x_component, guchar *device_y_component);
__device__ void gpu_grad_matrix_op(int iy, int ix, guchar *x_component, guchar *y_component, int width, float *xx_grad, float *xy_grad, float *yy_grad);
__device__ void gpu_blur_op(int iy, int ix, float *device_image, int width, int height);
__device__ void gpu_harris_op(int iy, int ix, int width, int height, float *harris, float *xx_grad, float *xy_grad, float *yy_grad);
__device__ void gpu_nonmax_op(int iy, int ix, int width, int height, float *harris);
__device__ void gpu_threshold_op(int iy, int ix, guchar *device_image, int width, int height, float *harris);

__global__ void main_kernel(guchar *device_image, int width, int height, guchar *device_x_component, guchar *device_y_component, float *xx_grad, float *xy_grad, float *yy_grad, float *harris)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int x, y;

	x = threadIdx.x + RADIUS;
	y = threadIdx.y + RADIUS;

	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		temp[y-RADIUS][x-RADIUS] = device_image[(iy-RADIUS) * width + (ix-RADIUS)];
		//temp[y-RADIUS][x+BLOCKSIZE] = device_image[(iy-RADIUS) * width + (ix+BLOCKSIZE)];
		//temp[y+BLOCKSIZE][x-RADIUS] = device_image[(iy+BLOCKSIZE) * width + (ix-RADIUS)];
		//temp[y+BLOCKSIZE][x+BLOCKSIZE] = device_image[(iy+BLOCKSIZE) * width + (ix+BLOCKSIZE)];		
		temp[y][x-RADIUS] = device_image[iy * width + (ix-RADIUS)];
		temp[y-RADIUS][x] = device_image[(iy-RADIUS) * width + ix];
	}
	else if(threadIdx.x == BLOCKSIZE-1 && threadIdx.y == 0)
	{
		temp[y+RADIUS][x+RADIUS] = device_image[(iy+RADIUS) * width + (ix+RADIUS)];
		temp[y][x+RADIUS] = device_image[iy * width + (ix+RADIUS)];
		temp[y-RADIUS][x] = device_image[(iy-RADIUS) * width + ix];
	}
	else if(threadIdx.x == 0 && threadIdx.y == BLOCKSIZE-1)
	{
		temp[y+RADIUS][x-RADIUS] = device_image[(iy+RADIUS) * width + (ix-RADIUS)];
		temp[y][x-RADIUS] = device_image[iy * width + (ix-RADIUS)];
		temp[y+RADIUS][x] = device_image[(iy+RADIUS) * width + ix];
	}
	else if(threadIdx.x == BLOCKSIZE-1 && threadIdx.y == BLOCKSIZE-1)
	{
		temp[y+RADIUS][x+RADIUS] = device_image[(iy+RADIUS) * width + (ix+RADIUS)];
		temp[y][x+RADIUS] = device_image[iy * width + (ix+RADIUS)];
		temp[y+RADIUS][x] = device_image[(iy+RADIUS) * width + ix];
	}
	else if(threadIdx.x < BLOCKSIZE-1 && threadIdx.y == 0)
	{
		temp[y-RADIUS][x] = device_image[(iy-RADIUS) * width + ix];
		temp[y+BLOCKSIZE][x] = device_image[(iy+BLOCKSIZE) * width + ix];
	}
	else if(threadIdx.x == 0 && threadIdx.y < BLOCKSIZE-1)
	{
		temp[y][x-RADIUS] = device_image[iy * width + (ix-RADIUS)];
		temp[y][x+BLOCKSIZE] = device_image[iy * width + (ix+BLOCKSIZE)];
	}
	/*else if(threadIdx.x == BLOCKSIZE-1 && threadIdx.y < BLOCKSIZE-1)
	{
		temp[y][x+RADIUS] = device_image[iy * width + (ix+RADIUS)];
	}
	else if(threadIdx.x < BLOCKSIZE-1 && threadIdx.y == BLOCKSIZE-1)
	{
		temp[y+RADIUS][x] = device_image[(iy+RADIUS) * width + ix];
	}*/

	temp[y][x] = device_image[iy * width + ix];

	__syncthreads();

	gpu_sobel_op(y, x, iy, ix, width, device_x_component, device_y_component);
	gpu_grad_matrix_op(iy, ix, device_x_component, device_y_component, width, xx_grad, xy_grad, yy_grad);

	gpu_blur_op(iy, ix, xx_grad, width, height);
	gpu_blur_op(iy, ix, xy_grad, width, height);
	gpu_blur_op(iy, ix, yy_grad, width, height);

	gpu_harris_op(iy, ix, width, height, harris, xx_grad, xy_grad, yy_grad);

    gpu_nonmax_op(iy, ix, width, height, harris);

    gpu_threshold_op(iy, ix, device_image, width, height, harris);
}


__device__ void gpu_sobel_op(int y, int x, const int iy, const int ix, int width, guchar *device_x_component, guchar *device_y_component)
{

	int r, c;

	int sumx = 0;
	int sumy = 0;

	for(r = -RADIUS; r <= RADIUS; r++)
	{
		for(c = -RADIUS; c <= RADIUS; c++)
		{
			sumx += __mul24(temp[y+r][x+c], gx[r + RADIUS][c + RADIUS]);
			sumy += __mul24(temp[y+r][x+c], gy[r + RADIUS][c + RADIUS]);
		}
	}
	
	device_x_component[iy * width + ix] = sumx;
	device_y_component[iy * width + ix] = sumy;

}


__device__ void gpu_blur_op(int iy, int ix, float *device_image, int width, int height)
{
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
		
	device_image[iy * width + ix] = __fdividef(sum, n);
}

__device__ void gpu_grad_matrix_op(int iy, int ix, guchar *x_component, guchar *y_component, int width, float *xx_grad, float *xy_grad, float *yy_grad)
{
	xx_grad[iy * width + ix] = x_component[iy * width + ix] * x_component[iy * width + ix];
	xy_grad[iy * width + ix] = x_component[iy * width + ix] * y_component[iy * width + ix];
	yy_grad[iy * width + ix] = y_component[iy * width + ix] * y_component[iy * width + ix];
}

__device__ void gpu_harris_op(int iy, int ix, int width, int height, float *harris, float *xx_grad, float *xy_grad, float *yy_grad)
{
	float det = xx_grad[iy * width + ix] * yy_grad[iy * width + ix] - xy_grad[iy * width + ix] * xy_grad[iy * width + ix];
	float trace = xx_grad[iy * width + ix] + yy_grad[iy * width + ix];

	harris[iy * width + ix] = det - 0.06 * trace * trace;
}

__device__ void gpu_nonmax_op(int iy, int ix, int width, int height, float *harris)
{
	int r, c;
	
	float curr = harris[iy * width + ix];

	int rStart = MAX(0, iy-1);
	int rEnd = MIN(height-1,iy+1);
	int cStart = MAX(0, ix-1);
	int cEnd = MIN(width-1, ix+1);
   
	for(r = rStart; r <= rEnd; r++)
	{
		for(c = cStart; c <= cEnd; c++)
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

__device__ void gpu_threshold_op(int iy, int ix, guchar *device_image, int width, int height, float *harris)
{
	if(harris[iy * width + ix] > THRESH)// && harris[iy * width + ix] < 1000000000.0)
	{
		device_image[iy * width + ix] = (guchar)255;
	}
}

