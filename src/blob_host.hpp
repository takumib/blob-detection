#ifndef __BLOB_HOST_HPP__
#define __BLOB_HOST_HPP__

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//#include <cuda.h>

#define BLOCKSIZE 8
#define MAX_SIGMA 32
#define THRESH 0.05

#define PI 3.141592653f
 
#define MAX(a, b) (((a)<(b))?(b):(a))
#define MIN(a, b) (((a)<(b))?(a):(b))

typedef unsigned char guchar;

extern void wakeup(void);
extern void run_gpu(guchar *host_image, const int width, const int height);
extern void tex_run_gpu(guchar *host_image, const int width, const int height);
extern void run_gpu_shared_op(guchar *host_image, const int width, const int height);
extern int has_cuda_device(void);

#endif
