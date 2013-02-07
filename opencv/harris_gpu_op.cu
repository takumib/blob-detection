#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cutil.h>

//#include <libgimp/gimp.h>
//#include <libgimp/gimpui.h>

//#include "gimp_main.hpp"
//#include "gimp_gui.hpp"

#include "harris_device_op.cu"
#include "harris_host.hpp"

#define BLOCKSIZE 8

typedef char guchar;

/*int has_cuda_device(void) 
{
	int device_count;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));

	return device_count;
}*/

/*void wakeup()
{
	 cudaEvent_t wakeGPU;
   	 CUDA_SAFE_CALL(cudaEventCreate(&wakeGPU));
}*/

void run_gpu_op(unsigned char *host_image, const int width, const int height)
{

	/** Our start and end times for our entire program **/
	//timeval start, end;

	/** the elapsed time from start to end **/
    //double elapsedtime;

	/** creates a timestamp of the current time **/
	//gettimeofday(&start, NULL);
	
	/** size of our image **/
	unsigned int mem_size;
	
	/** stores our gradient angles for non-maximum suppression **/
	unsigned int float_mem_size;

	/** our device image and our x and y gradient images **/
	guchar *device_image;
	guchar *device_img;
	guchar *device_x_component;
	guchar *device_y_component;
    guchar *buff;
	
	float *xx_grad;
	float *xy_grad;
	float *yy_grad;

	float *harris;
	float *harris_host = (float *)malloc(sizeof(float) * width * height);
	
	/** some important properties of the GPU **/
	cudaDeviceProp devProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&devProp, 0));

	/** amount of memory free and total memory on the GPU **/
	size_t mem_free, mem_total;

	/** we now want to load the memory information about the GPU **/ 
	CUDA_SAFE_CALL(cudaMemGetInfo(&mem_free, &mem_total));

	/** size of image is going to be our width * height unsigned chars **/ 
	mem_size = (sizeof(guchar) * width * height);

	/** our angles size is going to be the same except for the size of a float **/
	float_mem_size = (sizeof(float) * width * height);

	/** if we do not have enough memory left, then we must exit our program **/	
	if (mem_free < ((mem_size * 3) + (float_mem_size * 4) + 10)) 
	{
		//g_message("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		
		printf("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		return;
	}

	/** we now allocate memory for the device image and the x and y gradients on the GPU **/
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_image, mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_img, mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_x_component, mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_y_component, mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &buff, mem_size));

	CUDA_SAFE_CALL(cudaMalloc((void**) &xx_grad, float_mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &xy_grad, float_mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &yy_grad, float_mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &harris, float_mem_size));

	/** we now copy our image to the device image (dest, src, size, copy type) **/	
	CUDA_SAFE_CALL(cudaMemcpy(device_image, host_image, mem_size, cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(device_img, host_image, mem_size, cudaMemcpyHostToDevice));
	/** our amount of shared memory per block to use **/
	size_t shared_mem = 512;

	/** prints out the block dimensions **/
	//printf("%d %d", (int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	
	dim3 dimGrid((int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

	int shared_size = 16;

	dim3 dimGridShared((int)ceil((float)(width)/shared_size), (int)ceil((float)(height)/shared_size));
	dim3 dimBlockShared(shared_size, shared_size);

	/** call our gaussian blur kernel **/
	//gpu_average_blur<<<dimGridShared, dimBlockShared, shared_mem>>>(device_image, width, height);

	/** copies our blurred image from our device **/
	CUDA_SAFE_CALL(cudaMemcpy(device_x_component, device_image, mem_size, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_y_component, device_image, mem_size, cudaMemcpyDeviceToDevice));

	//gpu_compute_xgrad_row<<<dimGrid, dimBlock, shared_mem>>>(device_image, width, height, device_x_component);
	//gpu_compute_xgrad_col<<<dimGrid, dimBlock, shared_mem>>>(buff, width, height, device_x_component);
	//gpu_compute_xgrad_op<<<dimGridShared, dimBlockShared, shared_mem>>>(device_image, width, height, device_x_component, device_y_component);
	//gpu_compute_ygrad_op<<<dimGrid, dimBlock, shared_mem>>>(device_image, width, height, device_y_component);
	//gpu_compute_ygrad_row<<<dimGrid, dimBlock, shared_mem>>>(device_x_component, width, height, device_y_component);
	//gpu_compute_ygrad_col<<<dimGrid, dimBlock, shared_mem>>>(buff, width, height, device_y_component);

	main_kernel<<<dimGridShared, dimBlockShared, shared_mem>>>(device_image, width, height, device_x_component, device_y_component, xx_grad, xy_grad, yy_grad, harris);
	
	//gpu_gradient_matrix_op<<<dimGrid, dimBlock>>>(device_x_component, device_y_component, width, height, xx_grad, xy_grad, yy_grad);

	//gpu_average_blur_op<<<dimGridShared, dimBlockShared, shared_mem>>>(xx_grad, width, height);
	//gpu_average_blur_op<<<dimGridShared, dimBlockShared, shared_mem>>>(xy_grad, width, height);
	//gpu_average_blur_op<<<dimGridShared, dimBlockShared, shared_mem>>>(yy_grad, width, height);


	//float *xx_grad_host = (float *)malloc(width * height * sizeof(float));

	//gpu_harris_corner_op<<<dimGrid, dimBlock>>>(width, height, harris, xx_grad, xy_grad, yy_grad);

	//gpu_nonmax_suppression_op<<<dimGrid, dimBlock>>>(width, height, harris);
	
	//gpu_threshold_op<<<dimGrid, dimBlock>>>(device_img, width, height, harris);
	
	/*cudaMemcpy(xx_grad_host, harris, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	int i,j;

	for(i = 0; i  < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			printf("%.2f ", xx_grad_host[i * width + j]);
		}
		printf("\n");
	}*/
	
	/** copy our finalized image back to the host image **/
	cudaMemcpy(host_image, device_image, mem_size, cudaMemcpyDeviceToHost);

	/** free up the memory allocated on the GPU **/
	cudaFree(device_image);
	cudaFree(device_img);
	cudaFree(device_x_component);
	cudaFree(device_y_component);
	
	cudaFree(xx_grad);
	cudaFree(xy_grad);
	cudaFree(yy_grad);
	cudaFree(harris);
}

