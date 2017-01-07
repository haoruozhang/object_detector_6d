/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Function for generating normals
 * -- not used in latest version --
 */
#ifndef SURFACE_NORMALS_H
#define SURFACE_NORMALS_H

#include <cuda/cuda_utils.h>

namespace surface_normals_gpu{

    void generate_normals(const std::vector<unsigned short> &depthmap, int img_width, int img_height, float focal_length, std::vector<float> &host_normals);

}

#endif
