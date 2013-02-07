#ifndef __BLOB_COMMON_H__
#define __BLOB_COMMON_H__

#define MAX(a, b) (((a)<(b))?(b):(a))
#define MIN(a, b) (((a)<(b))?(a):(b))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <glib.h>

#define PI 3.141592653f
#define THRESH 0.05f
#define MAX_SIGMA 32

#define int_round(n) (floor(n + 0.5)) 
#define NGAUSSIANS ((int)((float)log(MAX_SIGMA) / ((float)log(sqrt(2)))) + 1)

void normalize(int width, int height, guchar *src, float *dest);
void compute_gaussians(int width, int height, float *img, float **gaussians);

#endif
