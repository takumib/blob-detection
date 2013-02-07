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

#include "harris_device_shared_op.cu"
#include "harris_host.hpp"

#define BLOCKSIZE 8

typedef char guchar;

void run_gpu_shared_op(unsigned char *host_image, const int width, const int height)
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
	//float_mem_size = (sizeof(float) * width * height);

	/** if we do not have enough memory left, then we must exit our program **/	
	if (mem_free < ((mem_size * 3) + 10)) 
	{
		//g_message("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		
		printf("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
		return;
	}

	//cudaEvent_t cts, cte;
	//cudaEventCreate(&cts);
	//cudaEventCreate(&cte);

	//float time;
	//cudaEventRecord(cts, 0);

	/** we now allocate memory for the device image and the x and y gradients on the GPU **/
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_image, mem_size));

	//cudaEventRecord(cte, 0);
	//cudaEventSynchronize(cte);

	//cudaEventElapsedTime(&time, cts, cte);

	//printf("Time to allocate memory on GPU: %.2f\n", time);

	//cudaEventRecord(cts, 0);
	/** we now copy our image to the device image (dest, src, size, copy type) **/	
	cudaStream_t stream1, stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	CUDA_SAFE_CALL(cudaMemcpyAsync(device_image, host_image, mem_size, cudaMemcpyHostToDevice, stream1));

	//cudaEventRecord(cte, 0);
	//cudaEventSynchronize(cte);

	//cudaEventElapsedTime(&time, cts, cte);

	//printf("Time to copy memory to GPU: %.2f\n", time);
	
	size_t shared_mem = 2048;

	dim3 dimGrid((int)ceil((float)(width)/BLOCKSIZE), (int)ceil((float)(height)/BLOCKSIZE));
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

	int shared_size = 16;

	dim3 dimGridShared((int)ceil((float)(width)/shared_size), (int)ceil((float)(height)/shared_size));
	dim3 dimBlockShared(shared_size, shared_size);

	//cudaEventRecord(cts, 0);

	kernel_shared_op<<<dimGridShared, dimBlockShared, shared_mem, stream2>>>(device_image, width, height);
	
	//cudaEventRecord(cte, 0);
	//cudaEventSynchronize(cte);

	//cudaEventElapsedTime(&time, cts, cte);

	//printf("Time for Kernel: %.2f\n", time);

	//cudaEventRecord(cts, 0);	
	/** copy our finalized image back to the host image **/
	cudaMemcpy(host_image, device_image, mem_size, cudaMemcpyDeviceToHost);

	//cudaEventRecord(cte, 0);
	//cudaEventSynchronize(cte);

	//cudaEventElapsedTime(&time, cts, cte);

	//printf("Time to copy back to host: %.2f\n", time);
	/** free up the memory allocated on the GPU **/
	cudaFree(device_image);
}

