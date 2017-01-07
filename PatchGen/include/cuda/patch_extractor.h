/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Extract patches from image using CUDA textures
 */
#ifndef PATCH_EXTRACTOR_H
#define PATCH_EXTRACTOR_H

#include <cuda/cuda_utils.h>
#include <stdio.h>
#include <vector>
#include <iostream>

namespace patch_extractor_gpu{

void extract_patches(const std::vector<float> &texture_3D,
                     int img_width,
                     int img_height,
                     int patch_size_in_voxels,
                     float voxel_size_in_m,
                     int stride_in_pixels,
                     float focal_length,
                     std::vector<float> &host_patches,
                     std::vector<int> &patches_loc,
                     bool generate_random_values, float distance_threshold);


void extract_patches_rgbd(const std::vector<float> &texture_3D,
                     int img_width,
                     int img_height,
                     int patch_size_in_voxels,
                     float voxel_size_in_m,
                     float max_depth_range_in_m,
                     int stride_in_pixels,
                     float percent,
                     float focal_length,
                     std::vector<float>& host_patches,
                     std::vector<int>& patches_loc,
                     bool generate_random_values,
                     float distance_threshold);


}

#endif
