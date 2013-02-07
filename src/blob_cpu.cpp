/*
 * Copyright (C) 2008, 2009, 2010 Richard Membarth <richard.membarth@cs.fau.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with NVIDIA CUDA Software Development Kit (or a modified version of that
 * library), containing parts covered by the terms of a royalty-free,
 * non-exclusive license, the licensors of this Program grant you additional
 * permission to convey the resulting work.
 */

#include "blob_cpu.hpp"
#include "blob_definitions.h"
void run_cpu(guchar *image, int width, int height, int channels) 
{
	//store a temporary image
	float *img = (float *)malloc(width * height * sizeof(float));
	
	float **gaussians = (float **)malloc(NGAUSSIANS * sizeof(float *));
	float **dog = (float **)malloc((NGAUSSIANS - 1) * sizeof(float *));
	float *features = (float *)malloc(width * height * sizeof(float));

	printf("Number of gaussians: %d\n", NGAUSSIANS);

	//normalize the source image and store it into our temp image
	normalize(width, height, image, img);

	//for each sigma value compute gaussian blur
	compute_gaussians(width, height, img, gaussians);

	printf("Computing DOG...\n");

	//perform our gaussian difference operation
	compute_dog(width, height, gaussians, dog);

	printf("Computing Greatest Difference...\n");

	compute_features(width, height, dog, features);

	printf("Drawing Squares ...\n");
	
	//draw our blobs on the image
	draw_blobs(width, height, features, image);

	int i;

	//deallocate our matrices
	for(i = 0; i < NGAUSSIANS; i++)
	{
		free(gaussians[i]);
	}

	for(i = 0; i < NGAUSSIANS-1; i++)
	{
		free(dog[i]);
	}

	free(img);
	free(gaussians);
	free(dog);
	free(features);
}

void normalize(int width, int height, guchar *src, float *dest)
{
	int i, j;

	//normalize our image scaled from 255
	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			dest[i * width + j] = (float)src[i * width + j] / (float)255;
		}
	}
}

void compute_gaussians(int width, int height, float *img, float **gaussians)
{
	int i;

	//for each sigma, perform a gaussian blur
	for(i = 0; i < NGAUSSIANS; i++)
	{
		float sigma = exp(log(sqrt(2)) * (float)((i)));

		gaussians[i] = (float *)malloc(width * height * sizeof(float));

		gaussian_blur(width, height, sigma, img, gaussians[i]);
	}
}

void compute_dog(int width, int height, float **gaussians, float **dog)
{
	int i;

	//for each pair of gaussians find their difference
	for(i = 0; i < NGAUSSIANS-1; i++)
	{
		dog[i] = (float *)malloc(width * height * sizeof(float));

		gaussian_diff(width, height, gaussians[i], gaussians[i+1], dog[i]);
	}
}

void compute_features(int width, int height, float **dog, float *features)
{
	int i, j, k, index;
	float greatest;

	for(i = 2; i < height - 2; i++)
	{
		for(j = 2; j < width - 2; j++)
		{
			index = 1;

			greatest = 0.0;

			//for a given pixel, find maximum difference of gaussian
			for(k = 1; k < NGAUSSIANS-1; k++)
			{
				if(dog[k][i * width + j] > greatest)
				{
					index = k;
					greatest = dog[k][i * width + j];
				}
			}

			//if the greatest difference of gaussian is greater than the threshold, then it is a blob
			if(greatest > THRESH)
			{
				if(greatestSpatially(width, height, i, j, dog[index]))
				{
					features[i * width + j] = (float)exp(log(sqrt(2)) * (float)index);
				}
			} 
			
		}
	}
}

void draw_blobs(int width, int height, float *features, guchar *image)
{
	int i, j;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			if(features[i * width + j] > 0.0)
			{
				drawSquare(width, height, i, j, int_round(features[i * width + j]), image);
			}
		}

	}
}

void gaussian_blur(int width, int height, float sigma, float *src, float *dest)
{
	int i, j;
	int row, col;

	float sum = 0.0;

	int radius = int_round(3 * sigma);

	//printf("radius: %d\n", radius);

	int kernelSize = 2 * radius + 1;

	float weights[kernelSize][kernelSize];
 
	float coeff = 1.0 / (2 * PI * sigma * sigma);	
	float sigma2 = -1.0 / (2 * sigma * sigma);

	//find the sum of the weights in a given kernel size
	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			weights[i+radius][j+radius] = (float)(coeff * exp(((i * i) + (j * j)) * sigma2));
			sum += weights[i+radius][j+radius];
		}
	}

	//find the mean of the weights based on the kernel size
	for(i = -radius; i <= radius; i++)
	{
		for(j = -radius; j <= radius; j++)
		{
			weights[i+radius][j+radius] = weights[i+radius][j+radius] / sum;
		}
	}

	//perform gaussian blur based on computed weights
	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			sum = 0.0;

			for(row = -radius; row <= radius; row++)
			{
				for(col = -radius; col <= radius; col++)
				{
					int rowidx = i+row < 0 ? abs(i+row) : i+row;
					int colidx = j+col < 0 ? abs(j+col) : j+col;
 
					sum += weights[row+radius][col+radius] * src[rowidx * width + colidx];
				}
			}
			
			dest[i * width + j] = sum;
		}
	}

}

void gaussian_diff(int width, int height, float *g1, float *g2, float *dest)
{
	int i, j;
	
	//find the difference of two gaussian blurs
	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			dest[i * width + j] = g2[i * width + j] - g1[i * width + j];
		}
	}
}

int greatestSpatially(int width, int height, int i, int j, float *dog)
{
	int row, col;

	float val = dog[i * width + j];

	int ret = 1;

	//if our current pixel is not the max in a window then return false
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

void drawSquare(int width, int height, int i, int j, int s, guchar *dest)
{
	int row, col;

	int l = MAX(0, j-s);
	int r = MIN(width-1, j+s);
	int t = MAX(0, i-s);
	int b = MIN(height-1, i+s);

	//draw left and right side of the square
	for(col = l; col < r; col++)
	{
		dest[t * width + col] = 0;
		dest[b * width + col] = 0; 
	}

	//draw top and bottom of the square
	for(row = t; row < b; row++)
	{
		dest[row * width + l] = 0;
		dest[row * width + r] = 0;
	}
}
