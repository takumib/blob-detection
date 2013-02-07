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

#ifndef __BLOB_CPU_HPP__
#define __BLOB_CPU_HPP__

#include "blob_common.h"


void compute_dog(int width, int height, float **gaussians, float **dog);
void compute_features(int width, int height, float **dog, float *features);
void draw_blobs(int width, int height, float *features, guchar *image);
void gaussian_blur(int width, int height, float sigma, float *src, float *dest);
void gaussian_diff(int width, int height, float *g1, float *g2, float *dest);
int greatestSpatially(int width, int height, int i, int j, float *dog);
void drawSquare(int width, int height, int i, int j, int s, guchar *dest);

#endif /* __BLOB_CPU_HPP__ */

