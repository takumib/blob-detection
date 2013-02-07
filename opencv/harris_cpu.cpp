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

#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

//#include "gimp_main.hpp"
//#include "gimp_gui.hpp"

#define MAX(a, b) (((a)<(b))?(b):(a))
#define MIN(a, b) (((a)<(b))?(a):(b))
#define THRESH 70000.0

//#ifdef PRINT_TIMES
#include <cutil.h>
//#endif

void average_blur(int width, int height, float *image);

void compute_xgrad(int width, int height, char *x_component, char *image);
void compute_ygrad(int width, int height, char *y_component, char *image);

void gradient_matrix(int width, int height, char *x_component, char *y_component, float *xx_grad, float *xy_grad, float *yy_grad);

void harris_corner(int width, int height, float *harris, float *xx_grad, float *xy_grad, float *yy_grad);

void nonmax_suppression(int width, int height, float *harris);

void threshold(int width, int height, guchar *image, float *harris);

void run_cpu(guchar *image, int width, int height, int channels) 
{
	char *img = (char *)malloc(width * height * sizeof(char));

	memcpy(img, image, (size_t)(width * height * sizeof(char)));

	//average_blur(width, height, img);

	char *x_component = (char *)malloc(width * height * sizeof(char));
 	char *y_component = (char *)malloc(width * height * sizeof(char));

	float *xx_grad = (float *)malloc(width * height * sizeof(float));
	float *xy_grad = (float *)malloc(width * height * sizeof(float));
	float *yy_grad = (float *)malloc(width * height * sizeof(float));

	float *harris = (float *)malloc(width * height * sizeof(float));

	memcpy(x_component, image, (size_t)(width * height * sizeof(char)));
	memcpy(y_component, image, (size_t)(width * height * sizeof(char)));

	compute_xgrad(width, height, x_component, img);
	compute_ygrad(width, height, y_component, img);

	gradient_matrix(width, height, x_component, y_component, xx_grad, xy_grad, yy_grad);

	average_blur(width, height, xx_grad);
	average_blur(width, height, xy_grad);
	average_blur(width, height, yy_grad);

	harris_corner(width, height, harris, xx_grad, xy_grad, yy_grad);

	nonmax_suppression(width, height, harris);

	threshold(width, height, image, harris);
}

void average_blur(int width, int height, float *image)
{
	int x, y,sum;
	int i, j;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			sum = 0;
			int count = 0;
			for (y = MAX(0, i-1); y <= MIN(height-1, i+1); y++) 
			{
				for (x = MAX(0,j-1); x <= MIN(width - 1, j+1); x++) 
				{

					sum += image[y * width + x]; // top left
					count++;
				}
			}
			
			image[i * width + j] = sum / count;  // Average the values for the blur 
			
			//image[i * width + j] = exp(-((float)(i*i + j*j)/2.0));

		}
	}
}

void compute_xgrad(int width, int height, char *x_component, char *image)
{
	int x, y, sum;
	int i, j, r, c;
	
	int gx[3][3] = { {-1, 0, 1},
					 {-2, 0, 2},
					 {-1, 0, 1} };

	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			sum = 0;

			for(r = MAX(0, y-1), i = 0; r <= MIN(height-1,y+1); r++, i++)
			{
				for(c = MAX(0, x-1), j=0; c <= MIN(width-1, x+1); c++, j++)
				{
					sum += image[r * width + c] * gx[i][j];
				}
			}

			x_component[y * width + x] = sum;	
		}
	}
}

void compute_ygrad(int width, int height, char *y_component, char *image)
{
	int x, y, r, c, i, j, sum;

	int gy[3][3] = { {-1, -2, -1},
					 { 0,  0,  0},
					 { 1,  2,  1} };

	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			sum = 0;

			for(r = MAX(0, y-1), i = 0; r <= MIN(height-1,y+1); r++, i++)
			{
				for(c = MAX(0, x-1), j=0; c <= MIN(width-1, x+1); c++, j++)
				{
					sum += image[r * width + c] * gy[i][j];
				}
			}

			y_component[y * width + x] = sum;	
		}
	}
}

void gradient_matrix(int width, int height, char *x_component, char *y_component, float *xx_grad, float *xy_grad, float *yy_grad)
{
	int i, j;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			xx_grad[i * width + j] = x_component[i * width + j] * x_component[i * width + j];
			xy_grad[i * width + j] = x_component[i * width + j] * y_component[i * width + j];
			yy_grad[i * width + j] = y_component[i * width + j] * y_component[i * width + j];

		}
	}
}

void harris_corner(int width, int height, float *harris, float *xx_grad, float *xy_grad, float *yy_grad)
{
	int i, j;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			float det = xx_grad[i * width + j] * yy_grad[i * width + j] - (xy_grad[i * width + j] * xy_grad[i * width + j]);
			float trace = xx_grad[i * width + j] + yy_grad[i * width + j];

			harris[i * width + j] = det - 0.04 * (trace * trace);
			//printf("%.2f ", harris[i * width + j]);
		}

		//printf("\n");
		//printf("\n");
	}
}


void nonmax_suppression(int width, int height, float *harris)
{
	int i, j, x, y;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			float curr = harris[i * width + j];

			if(harris[i * width + j] < 0.0)
			{
				harris[i * width + j] = 0.0;
			}

			for (y = MAX(0, i-1); y <= MIN(height-1, i+1); y++) 
			{
				for (x = MAX(0,j-1); x <= MIN(width-1, j+1); x++) 
				{
					if(curr > harris[y * width + x])
					{
						harris[y * width + x] = 0;
					}
					else
					{
						curr = harris[y * width + x];
					}
				}
			}

		}

	}
}


void threshold(int width, int height, guchar *image, float *harris)
{
	int i, j;

	for(i = 0; i < height; i++)
	{
		for(j = 0; j < width; j++)
		{
			if(harris[i * width + j] > THRESH)
			{
				image[i * width + j] = 255;
			}
		}

	}

}
