#include <cutil.h>
#include <stdio.h>
#include <stdlib.h>

#define THRESH 70000
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define RADIUS 1
#define BLOCKSIZE 16

#define ROWS (BLOCKSIZE + 2 * RADIUS)
#define COLS (BLOCKSIZE + 2 * RADIUS)

#define DIM (ROWS * COLS)

#define WIDTH (BLOCKSIZE + 2)
typedef char guchar;

/*__shared__ guchar sdevice_image[ROWS][COLS];
__shared__ guchar sdevice_x[ROWS][COLS];
__shared__ guchar sdevice_y[ROWS][COLS];

__shared__ float sxx_grad[ROWS][COLS];
__shared__ float sxy_grad[ROWS][COLS];
__shared__ float syy_grad[ROWS][COLS];

__shared__ float sharris[ROWS][COLS];
*/

__shared__ guchar sdevice_image[DIM];
__shared__ guchar sdevice_x[DIM];
__shared__ guchar sdevice_y[DIM];

__shared__ float sxx_grad[DIM];
__shared__ float sxy_grad[DIM];
__shared__ float syy_grad[DIM];

__shared__ float sharris[DIM];

__device__ void gpu_sobel_shared_op(int y, int x);
__device__ void gpu_grad_matrix_shared_op(int y, int x);
__device__ void gpu_blur_shared_op(int y, int x, int rstart, int rend, int cstart, int cend, float device_image[DIM]);
__device__ void gpu_harris_shared_op(int y, int x);
__device__ void gpu_nonmax_shared_op(int y, int x, int rstart, int rend, int cstart, int cend);
__device__ void gpu_threshold_shared_op(int y, int x, const int index, guchar *device_image);

__global__ void kernel_shared_op(guchar *device_image, int width, int height)
{
	const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	const int index = __mul24(iy, width) + ix;

	const int x = threadIdx.x + RADIUS;
	const int y = threadIdx.y + RADIUS;

	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		sdevice_image[(y-RADIUS) * WIDTH + x-RADIUS] = device_image[(iy-RADIUS) * width + (ix-RADIUS)];
		sdevice_image[(y-RADIUS) * WIDTH + x+BLOCKSIZE] = device_image[(iy-RADIUS) * width + (ix+BLOCKSIZE)];
		sdevice_image[(y+BLOCKSIZE) * WIDTH + x-RADIUS] = device_image[(iy+BLOCKSIZE) * width + (ix-RADIUS)];
		sdevice_image[(y+BLOCKSIZE) * WIDTH + x+BLOCKSIZE] = device_image[(iy+BLOCKSIZE) * width + (ix+BLOCKSIZE)];		
	}
	if(threadIdx.x < BLOCKSIZE && threadIdx.y == 0)
	{
		sdevice_image[(y-RADIUS) * WIDTH + x] = device_image[(iy-RADIUS) * width + ix];
		sdevice_image[(y+BLOCKSIZE) * WIDTH + x] = device_image[(iy+BLOCKSIZE) * width + ix];
	}
	if(threadIdx.x == 0 && threadIdx.y < BLOCKSIZE)
	{
		sdevice_image[y * WIDTH + x-RADIUS] = device_image[iy * width + (ix-RADIUS)];
		sdevice_image[y * WIDTH + x+BLOCKSIZE] = device_image[iy * width + (ix+BLOCKSIZE)];
	}

	sdevice_image[y * WIDTH + x] = device_image[index];

	__syncthreads();

	gpu_sobel_shared_op(y, x);

	__syncthreads();

	gpu_grad_matrix_shared_op(y, x);

	//__syncthreads();
	/*int rstart = MAX(RADIUS, y-1);
	int rend = MIN(BLOCKSIZE, y+1);

	int cstart = MAX(RADIUS, x-1);
	int cend = MIN(BLOCKSIZE, x+1);
	*/

	int rstart = max(RADIUS, y-1);
	int rend = min(BLOCKSIZE, y+1);

	int cstart = max(RADIUS, x-1);
	int cend = min(BLOCKSIZE, x+1);

	gpu_blur_shared_op(y, x, rstart, rend, cstart, cend, sxx_grad);
	gpu_blur_shared_op(y, x, rstart, rend, cstart, cend, sxy_grad);
	gpu_blur_shared_op(y, x, rstart, rend, cstart, cend, syy_grad);

	//__syncthreads();

	gpu_harris_shared_op(y, x);

	//__syncthreads();

    gpu_nonmax_shared_op(y, x, rstart, rend, cstart, cend);

	//__syncthreads();

    gpu_threshold_shared_op(y, x, index, device_image);
}


__device__ void gpu_sobel_shared_op(int y, int x)
{
	/*int sumx = (-sdevice_image[y-1][x-1]) + (sdevice_image[y-1][x+1]) + 
			   (__mul24(-2, sdevice_image[y][x-1])) + (__mul24(2, sdevice_image[y][x+1])) + 
			   (-sdevice_image[y+1][x-1]) + (sdevice_image[y+1][x+1]);

	int sumy = (-sdevice_image[y-1][x-1]) + (__mul24(-2, sdevice_image[y-1][x])) + 
			   (-sdevice_image[y-1][x+1]) + (sdevice_image[y+1][x-1]) + 
			   (__mul24(2, sdevice_image[y+1][x])) + (sdevice_image[y+1][x+1]);
	  */
	sdevice_x[y * WIDTH + x] = (-sdevice_image[(y-1) * WIDTH + x-1]) + (sdevice_image[(y-1) * WIDTH + x+1]) + 
			   		  (__mul24(-2, sdevice_image[y * WIDTH + x-1])) + (__mul24(2, sdevice_image[y * WIDTH + x+1])) + 
			   		  (-sdevice_image[(y+1) * WIDTH + x-1]) + (sdevice_image[(y+1) * WIDTH + x+1]);
	sdevice_y[y * WIDTH + x] = (-sdevice_image[(y-1) * WIDTH + x-1]) + (__mul24(-2, sdevice_image[(y-1) * WIDTH + x])) + 
			   	  	  (-sdevice_image[(y-1) * WIDTH + x+1]) + (sdevice_image[(y+1) * WIDTH + x-1]) + 
			   		  (__mul24(2, sdevice_image[(y+1) * WIDTH + x])) + (sdevice_image[(y+1) * WIDTH + x+1]);
}


__device__ void gpu_blur_shared_op(int y, int x, int rstart, int rend, int cstart, int cend, float device_image[DIM])
{
	int r, c;

	int sum = 0;
	int n = 0;
	
	/*int rstart = MAX(RADIUS, y-1);
	int rend = MIN(BLOCKSIZE, y+1);

	int cstart = MAX(RADIUS, x-1);
	int cend = MIN(BLOCKSIZE, x+1);
	*/

	for(r = rstart; r <= rend; r++)
	{
		int row = r * WIDTH;
		for(c = cstart; c <= cend; c++)
		{
			sum += device_image[row + c];
			n++;
		}
	}

	device_image[y * WIDTH + x] =  __fdividef(sum, n);
}

__device__ void gpu_grad_matrix_shared_op(int y, int x)
{
	int index = y * WIDTH + x;

	sxx_grad[index] = __mul24(sdevice_x[index], sdevice_x[index]);
	sxy_grad[index] = __mul24(sdevice_x[index], sdevice_y[index]);
	syy_grad[index] = __mul24(sdevice_y[index], sdevice_y[index]);
}

__device__ void gpu_harris_shared_op(int y, int x)
{
	int index = y * WIDTH + x;

	float det = __mul24(sxx_grad[index], syy_grad[index]) - __mul24(sxy_grad[index], sxy_grad[index]);
	float trace = sxx_grad[index] + syy_grad[index];

	sharris[index] = det - 0.06 * __powf(trace, 2);
}

__device__ void gpu_nonmax_shared_op(int y, int x, int rstart, int rend, int cstart, int cend)
{
	int r, c;
	
	float curr = sharris[y * WIDTH + x];
	
	for(r = rstart; r <= rend; r++)
	{
		int row = r * WIDTH;
		for(c = cstart; c <= cend; c++)
		{
			if(curr > sharris[row + c])
			{
				sharris[row + c] = 0;
			}	
			else
			{	
				curr = sharris[row + c];
			}

		}
	}

}

__device__ void gpu_threshold_shared_op(int y, int x, const int index, guchar *device_image)
{
	device_image[index] = sharris[y * WIDTH + x] > THRESH ? (guchar)255 : device_image[index]; 
}

