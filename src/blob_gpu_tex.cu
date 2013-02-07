//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <math.h>
//#include <sys/time.h>
//#include <cutil.h>

//#include <cuda.h>
//#include <libgimp/gimp.h>
//#include <libgimp/gimpui.h>

//#include "gimp_main.hpp"
//#include "gimp_gui.hpp"

//#include "harris_device_tex.cu"
#include "blob_host.hpp"

//#define BLOCKSIZE 8

//#define MAX_SIGMA 32

//#define THRESH 0.05
//#define MAX(a, b) (((a) > (b)) ? (a) : (b))
//#define MIN(a, b) (((a) < (b)) ? (a) : (b))

//typedef char guchar;

/*int has_cuda_device(void) 
{
	int device_count;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));

	return device_count;
}

void wakeup()
{
	 cudaEvent_t wakeGPU;
   	 CUDA_SAFE_CALL(cudaEventCreate(&wakeGPU));
}*/

//#define PI 3.14159

texture<float> tex_img;

void tex_compute_weights(float sigma, int radius, int kernelSize, float *weights);
int tex_greatestSpatially(int width, int height, int i, int j, float *dog);
void tex_drawSquare(int width, int height, int i, int j, int s, unsigned char *dest);
int tex_gpu_int_round(float n);

__global__ void tex_gpu_gaussian_blur(int width, int height, float sigma, float *weights, float *src, float *dest);
__global__ void tex_gpu_gaussian_diff(int width, int height, float *g1, float *g2, float *dest);


void tex_run_gpu(unsigned char *host_image, const int width, const int height)
{
	/** size of our image **/
	size_t mem_size;
	
	/** stores our gradient angles for non-maximum suppression **/
	//unsigned int float_mem_size;

	float *h_img;
	float **h_gaussians;
	float *h_features;

	/** our device image and our x and y gradient images **/
	float *d_img;
	float *d_gaussians;
	float *d_dog;
	//float *d_features;	

	float *d_weights;
	
	/** some important properties of the GPU **/
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	/** amount of memory free and total memory on the GPU **/
	size_t mem_free, mem_total;

	/** we now want to load the memory information about the GPU **/ 
	cudaMemGetInfo(&mem_free, &mem_total);

	/** size of image is going to be our width * height unsigned chars **/ 
	mem_size = (sizeof(float) * width * height);

	/** if we do not have enough memory left, then we must exit our program **/	
	//if (mem_free < ((mem_size * 3) + (float_mem_size * 4) + 10)) 
	//{
		//g_message("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		
	//	printf("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
	//	return;
	//}

	int nGaussians = (int)((float)log(MAX_SIGMA) / (float)log(sqrt(2))) + 1;

	int i, j;

	/** we now allocate memory for the device image and the x and y gradients on the GPU **/
	cudaMalloc((void**) &d_img, mem_size);

	cudaBindTexture(NULL, tex_img, d_img, mem_size);

	cudaMalloc((void**) &d_gaussians, (width * height * sizeof(float)));

	h_img = (float *)malloc(width * height * sizeof(float));

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			h_img[i * width + j] = (float)host_image[i * width + j] / (float)255;
		}
	}


	/** we now copy our image to the device image (dest, src, size, copy type) **/	
	cudaMemcpy(d_img, h_img, mem_size, cudaMemcpyHostToDevice);

	printf("%d %d\n", (int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	
	dim3 blocks((int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	dim3 threads(BLOCKSIZE, BLOCKSIZE);

	h_gaussians = (float **)malloc(nGaussians * sizeof(float *));

	for(i = 0; i < nGaussians; i++)
	{
		float sigma = exp(log(sqrt(2)) * (float)(i));

		int radius = (int)floor((3 * sigma) + 0.5);

		int kernelSize = 2 * radius + 1;
	
 		float *h_weights = (float *)malloc(kernelSize * kernelSize * sizeof(float));
	
		tex_compute_weights(sigma, radius, kernelSize, h_weights);

		cudaMalloc((void**) &d_weights, kernelSize * kernelSize * sizeof(float));

		cudaMemcpy(d_weights, h_weights, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

		printf("radius: %d\n", radius);
		
		/** call our gaussian blur kernel **/
		tex_gpu_gaussian_blur<<<blocks, threads>>>(width, height, sigma, d_weights, d_img, d_gaussians);

		h_gaussians[i] = (float *)malloc(width * height * sizeof(float));

		cudaMemcpy(h_gaussians[i], d_gaussians, mem_size, cudaMemcpyDeviceToHost);

		cudaFree(d_weights);
	}

	float **h_dog = (float **)malloc((nGaussians - 1) * sizeof(float *));

	float *d_g1, *d_g2;

	cudaMalloc((void **) &d_dog, width * height * sizeof(float));
	cudaMalloc((void **) &d_g1, width * height * sizeof(float));
	cudaMalloc((void **) &d_g2, width * height * sizeof(float));
	
	for(i = 0; i < nGaussians - 1; i++)
	{
		h_dog[i] = (float *)malloc(width * height * sizeof(float));

		cudaMemcpy(d_g1, h_gaussians[i], mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_g2, h_gaussians[i+1], mem_size, cudaMemcpyHostToDevice);

		tex_gpu_gaussian_diff<<<blocks, threads>>>(width, height, d_g1, d_g2, d_dog);

		cudaMemcpy(h_dog[i], d_dog, mem_size, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_dog);
	cudaFree(d_g1);
	cudaFree(d_g2);

	h_features = (float *)malloc(width * height * sizeof(float));
	
	int k, index;
	float greatest;

	for(i = 2; i < height - 2; i++)
	{
		for(j = 2; j < width - 2; j++)
		{
			index = 1;

			greatest = 0.0;

			for(k = 1; k < nGaussians - 1; k++)
			{
				if(h_dog[k][i * width + j] > greatest)
				{
					index = k;
					greatest = h_dog[k][i * width + j];
				}
			}

			if(greatest > THRESH)
			{
				if(tex_greatestSpatially(width, height, i, j, h_dog[index]))
				{
					h_features[i * width + j] = (float)exp(log(sqrt(2)) * (float)index);
				}
			}			
		}
	}

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			if(h_features[i * width + j] > 0.0)
			{
				tex_drawSquare(width, height, i, j, tex_gpu_int_round(h_features[i * width + j]), host_image);
			}
		}
	}

	printf("Finished Kernel \n");

	/*for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			host_image[i * width + j] = (unsigned char)h_img[i * width + j];
		}
	}*/

	/** free up the memory allocated on the GPU **/
	cudaFree(d_img);
//	cudaFree(d_features);

	cudaFree(d_gaussians);
	
	//cudaFree(d_dog);

	for(i = 0; i < nGaussians; i++)
	{
		free(h_gaussians[i]);
	}

	for(i = 0; i < nGaussians - 1; i++)
	{
		free(h_dog[i]);
	}

	free(h_gaussians);
	free(h_dog);
	free(h_img);

}

int tex_gpu_int_round(float n)
{
	return (int)floor(n + 0.5);
}

void tex_compute_weights(float sigma, int radius, int kernelSize, float *weights)
{
	int i, j;

	float coeff = 1.0 / (2 * PI * sigma * sigma);
	float sigma2 = -1.0 / (2 * sigma * sigma);
	
	float sum = 0.0;

	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			weights[(i+radius) * kernelSize + (j+radius)] = (float)(coeff * exp(((i * i) + (j * j)) * sigma2));
			sum += weights[(i+radius) * kernelSize + (j+radius)];
		}
	}		

	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			weights[(i+radius) * kernelSize + (j+radius)] = weights[(i+radius) * kernelSize + (j+radius)] / sum;
		}
	}
}

int tex_greatestSpatially(int width, int height, int i, int j, float *dog)
{
	int row, col;

	float val = dog[i * width + j];

	int ret = 1;

	for(row = -2; row <= 2; row++)
	{
		for(col = -2; col <= 2; col++)
		{
			if(val != MAX(dog[(i+row) * width + (j+col)], val))
			{
				ret = 0;
			}
		}
	}

	return ret;
}

void tex_drawSquare(int width, int height, int i, int j, int s, unsigned char *dest)
{
	int row, col;

	int l = MAX(0, j-s);
	int r = MIN(width-1, j+s);
	int t = MAX(0, i-s);
	int b = MIN(height-1, i+s);

	for(col = l; col < r; col++)
	{
		dest[t * width + col] = 0;
		dest[b * width + col] = 0;
	}

	for(row = t; row < b; row++)
	{
		dest[row * width + l] = 0;
		dest[row * width + r] = 0;
	}
}

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
