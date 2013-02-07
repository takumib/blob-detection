#include "blob_common.h"
#include "blob_device.cu"
#include "blob_host.hpp"
#include "blob_gpu.hpp"

int has_cuda_device(void) 
{
	int device_count;

	cudaGetDeviceCount(&device_count);

	return device_count;
}

void wakeup()
{
	 cudaEvent_t wakeGPU;
 	 cudaEventCreate(&wakeGPU);
}


void run_gpu(unsigned char *host_image, const int width, const int height)
{
	/** size of our image **/
	size_t mem_size;
	
	/** our host matrices **/
	float *h_img;
	float **h_gaussians;
	float *h_features;

	/** our device matrices **/
	float *d_img;
	float *d_gaussians;
	float *d_dog;
	float *d_weights;
	
	/** gather gpu device information **/
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	/** amount of memory free and total memory on the GPU **/
	size_t mem_free, mem_total;

	/** we now want to load the memory information about the GPU **/ 
	cudaMemGetInfo(&mem_free, &mem_total);

	/** size of image is going to be our width * height unsigned chars **/ 
	mem_size = (sizeof(float) * width * height);

	/** if we do not have enough memory left, then we must exit our program **/	
	if (mem_free < (mem_size * 4)) 
	{
		printf("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		return;
	}

	int nGaussians = (int)((float)log(MAX_SIGMA) / (float)log(sqrt(2))) + 1;

	int i, j;

	/** allocate memory on the GPU for the image and gaussians **/
	cudaMalloc((void**) &d_img, mem_size);
	cudaMalloc((void**) &d_gaussians, (width * height * sizeof(float)));

	/** allocate memory for the temporary host image **/
	h_img = (float *)malloc(width * height * sizeof(float));

	/** store normalized image into host image **/
	/*for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			h_img[i * width + j] = (float)host_image[i * width + j] / (float)255;
		}
	}*/

	normalize(width, height, host_image, h_img);

	/** we now copy our normlized host image to the device image (dest, src, size, copy type) **/	
	cudaMemcpy(d_img, h_img, mem_size, cudaMemcpyHostToDevice);

	/** prints out number of threads per block in each dimension **/
	//printf("%d %d\n", (int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	
	/** number of blocks and threads respectively **/
	dim3 blocks((int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	dim3 threads(BLOCKSIZE, BLOCKSIZE);

	/** this is where we are going to store our host version of the gaussian blurs **/
	h_gaussians = (float **)malloc(nGaussians * sizeof(float *));

	/*for(i = 0; i < nGaussians; i++)
	{
		float sigma = exp(log(sqrt(2)) * (float)(i));

		int radius = (int)floor((3 * sigma) + 0.5);

		int kernelSize = 2 * radius + 1;
	
		// host version of weights
 		float *h_weights = (float *)malloc(kernelSize * kernelSize * sizeof(float));
	
		/// compute the weights based on the sigma value
		compute_weights(sigma, radius, kernelSize, h_weights);

		// allocate memory for the weights
		cudaMalloc((void**) &d_weights, kernelSize * kernelSize * sizeof(float));

		// weights to the GPU
		cudaMemcpy(d_weights, h_weights, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

		printf("radius: %d\n", radius);
		
		// call our gaussian blur kernel 
		gpu_gaussian_blur<<<blocks, threads>>>(width, height, sigma, d_weights, d_img, d_gaussians);

		// allocate memory for gaussian
		h_gaussians[i] = (float *)malloc(width * height * sizeof(float));

		// copy our gaussian blur from the GPU back to the host
		cudaMemcpy(h_gaussians[i], d_gaussians, mem_size, cudaMemcpyDeviceToHost);

		// free our current value of weights on the GPU
		cudaFree(d_weights);
	}
	*/
	
	compute_gaussians(width, height, h_img, h_gaussians);
	
	/** allocate memory for our difference of gaussians **/
	float **h_dog = (float **)malloc((nGaussians - 1) * sizeof(float *));

	float *d_g1, *d_g2;

	/** allocate GPU memory for our difference of gaussians **/
	cudaMalloc((void **) &d_dog, width * height * sizeof(float));
	cudaMalloc((void **) &d_g1, width * height * sizeof(float));
	cudaMalloc((void **) &d_g2, width * height * sizeof(float));
	
	for(i = 0; i < nGaussians - 1; i++)
	{
		/** allocate memory for each dog **/
		h_dog[i] = (float *)malloc(width * height * sizeof(float));

		/** copy each gaussian to GPU memory **/
		cudaMemcpy(d_g1, h_gaussians[i], mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_g2, h_gaussians[i+1], mem_size, cudaMemcpyHostToDevice);

		/** perform our gaussian difference **/
		gpu_gaussian_diff<<<blocks, threads>>>(width, height, d_g1, d_g2, d_dog);

		/** copy our results back to host memory **/
		cudaMemcpy(h_dog[i], d_dog, mem_size, cudaMemcpyDeviceToHost);
	}

	/** deallocate our dog, and our temporary gaussians **/
	cudaFree(d_dog);
	cudaFree(d_g1);
	cudaFree(d_g2);

	/** allocate memory for our blobs **/
	h_features = (float *)malloc(width * height * sizeof(float));
	
	/** finding our features **/
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
				if(greatestSpatially(width, height, i, j, h_dog[index]))
				{
					h_features[i * width + j] = (float)exp(log(sqrt(2)) * (float)index);
				}
			}			
		}
	}

	/** drawing our squares **/
	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			if(h_features[i * width + j] > 0.0)
			{
				gpu_drawSquare(width, height, i, j, int_round(h_features[i * width + j]), host_image);
			}
		}
	}

	printf("Finished Kernel \n");


	/** free up the memory allocated on the GPU **/
	cudaFree(d_img);
	cudaFree(d_gaussians);
	
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

void gaussian_blur(int width, int height, float sigma, float *d_img, float *h_gaussian)
{
	int i, j;

	float coeff = 1.0 / (2 * PI * sigma * sigma);
	float sigma2 = -1.0 / (2 * sigma * sigma);
	
	int radius = int_round(3 * sigma);
	int kernelSize = 2 * radius + 1;
 	
	float *h_weights = (float *)malloc(kernelSize * kernelSize * sizeof(float));
	float *d_weights;

	float *d_gaussian;
	
	float sum = 0.0;

	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			h_weights[(i+radius) * kernelSize + (j+radius)] = (float)(coeff * exp(((i * i) + (j * j)) * sigma2));
			sum += h_weights[(i+radius) * kernelSize + (j+radius)];
		}
	}		

	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			h_weights[(i+radius) * kernelSize + (j+radius)] = h_weights[(i+radius) * kernelSize + (j+radius)] / sum;
		}
	}

	dim3 blocks((int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	dim3 threads(BLOCKSIZE, BLOCKSIZE);

	/** allocate memory for the weights **/
	cudaMalloc((void**) &d_weights, kernelSize * kernelSize * sizeof(float));
	cudaMalloc((void**) &d_gaussian, (width * height * sizeof(float)));
	
	/** weights to the GPU **/
	cudaMemcpy(d_weights, h_weights, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
		
	/** call our gaussian blur kernel **/
	gpu_gaussian_blur<<<blocks, threads>>>(width, height, sigma, d_weights, d_img, d_gaussian);

	/** copy our gaussian blur from the GPU back to the host **/
	cudaMemcpy(h_gaussian, d_gaussian, (width * height * sizeof(float)), cudaMemcpyDeviceToHost);

	free(h_weights);

	/** free our current value of weights on the GPU **/
	cudaFree(d_weights);
	cudaFree(d_gaussian);
}

void compute_weights(float sigma, int radius, int kernelSize, float *weights)
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

int greatestSpatially(int width, int height, int i, int j, int s, guchar *dog)
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

void gpu_drawSquare(int width, int height, int i, int j, int s, guchar *dest)
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
