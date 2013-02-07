#ifndef __BLOB_GPU_HPP__
#define __BLOB_GPU_HPP__

void gaussian_blur(int width, int height, float sigma, float *d_img, float *h_gaussian);

void compute_weights(float sigma, int radius, int kernelSize, float *weights);
int greatestSpatially(int width, int height, int i, int j, float *dog);
void gpu_drawSquare(int width, int height, int i, int j, int s, unsigned char *dest);
int gpu_int_round(float n);

#endif
