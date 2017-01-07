#include <iostream>
#include <cuda/patch_extractor.h>
#include <cuda/cuda_utils.h>
#include <curand_kernel.h>


namespace patch_extractor_gpu {

//save B,G,R,D, (X,Y,Z) surface normals as 3D texture
texture<float, 3, cudaReadModeElementType> tex;

extern "C"
__global__
void extract(float *dev_patches, int *dev_patches_loc, int stride, int patch_size_in_voxels, float voxel_size_in_m, float focal_length,
             int img_width, int img_height, bool generate_random_values){

    __shared__ float r_rand, g_rand, b_rand, x_rand, y_rand, z_rand;

    if(threadIdx.x == 0){
        bool found = false;
        while(!found){
            unsigned int seed = static_cast<unsigned  int>( clock64() );
            curandState s;
            curand_init(seed, 0, 0, &s);
            r_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
            g_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
            b_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
            x_rand = static_cast<float>(curand(&s) % 100000) - 50000;
            y_rand = static_cast<float>(curand(&s) % 100000) - 50000;
            //Z can't be < 0. We assume Z < 0 has for the interior of a surface, which is not visible to us
            z_rand = static_cast<float>(curand(&s) % 50000);
            float norm = sqrt(x_rand*x_rand + y_rand*y_rand + z_rand*z_rand);
            if(norm != 0){
                x_rand /= norm;
                y_rand /= norm;
                z_rand /= norm;
                found = true;
            }
        }
    }
    __syncthreads();


    int patch_center_x = dev_patches_loc[blockIdx.x*2];
    int patch_center_y = dev_patches_loc[blockIdx.x*2 +1];

    //get depth of central pixel    
    float patch_center_d = tex3D(tex, 3 + 0.5f, patch_center_x + 0.5f, patch_center_y + 0.5f)/1000; //channel 4 (index 3) is Depth   

    // get patch size according to current depth
    // validity of patch_center_d and patch lying inside image already checked
    int adaptive_patch_size_in_pixels = static_cast<int> ( static_cast<float>(patch_size_in_voxels) * voxel_size_in_m / patch_center_d * focal_length );
    int patch_start_pos_x = patch_center_x - adaptive_patch_size_in_pixels/2;
    int patch_start_pos_y = patch_center_y - adaptive_patch_size_in_pixels/2;

    //cyclic assignment of pixels to threads
    int cur_patch_pixel = threadIdx.x;
    while(cur_patch_pixel < patch_size_in_voxels * patch_size_in_voxels){

        //get patch coordinates for the pixel
        int thread_x = cur_patch_pixel % patch_size_in_voxels;
        int thread_y = cur_patch_pixel / patch_size_in_voxels;
        //get output pixel offset
        int dev_patches_pos = blockIdx.x * patch_size_in_voxels * patch_size_in_voxels * 6 + thread_y * patch_size_in_voxels * 6 + thread_x * 6;

        //get interpolation coordinates for the pixel
        float texture_pos_x = static_cast<float>(patch_start_pos_x) + static_cast<float>(thread_x) * ( static_cast<float>(adaptive_patch_size_in_pixels) / static_cast<float>(patch_size_in_voxels) );
        float texture_pos_y = static_cast<float>(patch_start_pos_y) + static_cast<float>(thread_y) * ( static_cast<float>(adaptive_patch_size_in_pixels) / static_cast<float>(patch_size_in_voxels) );

        // get depth to see if it belongs to object       
        bool pixel_in_object = false;
        float d = tex3D(tex,  3 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f)/1000.0f;
        if(d > 0){
            float x = tex3D(tex,  4 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
            float y = tex3D(tex,  5 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
            float z = tex3D(tex,  6 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
            //normalize interpolated values
            float norm = sqrt(x*x + y*y + z*z);
            // check if x-y-z are not 0, i.e. surface normal exists or it background
            if(norm > 0){
                //get pixel value from texture interpolation
                dev_patches[dev_patches_pos + 0] = tex3D(tex,  0 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
                dev_patches[dev_patches_pos + 1] = tex3D(tex,  1 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
                dev_patches[dev_patches_pos + 2] = tex3D(tex,  2 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
                dev_patches[dev_patches_pos + 3] = x / norm;
                dev_patches[dev_patches_pos + 4] = y / norm;
                dev_patches[dev_patches_pos + 5] = z / norm;
                pixel_in_object = true;
            }
        }
        if(!pixel_in_object){
            if(generate_random_values){
                dev_patches[dev_patches_pos + 0] = b_rand;
                dev_patches[dev_patches_pos + 1] = g_rand;
                dev_patches[dev_patches_pos + 2] = r_rand;
                dev_patches[dev_patches_pos + 3] = x_rand;
                dev_patches[dev_patches_pos + 4] = y_rand;
                dev_patches[dev_patches_pos + 5] = z_rand;
            } else {
                dev_patches[dev_patches_pos + 0] = 0;
                dev_patches[dev_patches_pos + 1] = 0;
                dev_patches[dev_patches_pos + 2] = 0;
                dev_patches[dev_patches_pos + 3] = 0;
                dev_patches[dev_patches_pos + 4] = 0;
                dev_patches[dev_patches_pos + 5] = 0;
            }
        }

        cur_patch_pixel += blockDim.x;
    }
}

//texture_3D: IMG_HEIGHT x IMG_WIDTH x 7 (RGBDXYZ)
void extract_patches(const std::vector<float> &texture_3D,
                     int img_width,
                     int img_height,
                     int patch_size_in_voxels,
                     float voxel_size_in_m,
                     int stride_in_pixels,
                     float focal_length,
                     std::vector<float>& host_patches,
                     std::vector<int>& patches_loc,
                     bool generate_random_values,
                     float distance_threshold)
{  
    cuda_timer ct;
    //ct.start_timer();

    cuda_initializer cuda_config;
    cuda_config.deviceInit();
    //ct.print_timer("Cuda Init");

    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;


    cudaArray *dev_texture_arr;

    //first dimenstion is the most quickly changing one
    cudaExtent extend = make_cudaExtent(7, img_width, img_height);

    CUDA_SAFE_CALL( cudaMalloc3DArray(&dev_texture_arr, &tex.channelDesc, extend) );

    cudaMemcpy3DParms params = {0};    
    params.srcPtr = make_cudaPitchedPtr((void*)&texture_3D[0], 7 * sizeof(float), 7, img_width);
    params.dstArray = dev_texture_arr;
    params.extent = extend;
    params.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&params) );

    CUDA_SAFE_CALL( cudaBindTextureToArray(tex, dev_texture_arr, tex.channelDesc) );
    //ct.print_timer("Texture binding");


    int max_threads = patch_size_in_voxels * patch_size_in_voxels < cuda_config.deviceProp.maxThreadsPerBlock ?
        patch_size_in_voxels * patch_size_in_voxels : cuda_config.deviceProp.maxThreadsPerBlock;

    //get valid patch locations - [x1, y1, x2, y2 .. ]
    patches_loc.resize(0);
    for(int h=0; h<img_height; h += stride_in_pixels){
        for(int w=0; w<img_width; w += stride_in_pixels){
            int tex3d_depth_pos = h*img_width*7 + w*7 + 3;
            if(texture_3D[tex3d_depth_pos] != 0 && (float)texture_3D[tex3d_depth_pos]/1000.0f < distance_threshold){
                int adaptive_patch_size_in_pixels = static_cast<int> ( static_cast<float>(patch_size_in_voxels) * voxel_size_in_m / (texture_3D[tex3d_depth_pos]/1000.0f) * focal_length );
                int patch_start_pos_x = w - adaptive_patch_size_in_pixels/2;
                int patch_end_pos_x = patch_start_pos_x + adaptive_patch_size_in_pixels - 1;
                int patch_start_pos_y = h - adaptive_patch_size_in_pixels/2;
                int patch_end_pos_y = patch_start_pos_y + adaptive_patch_size_in_pixels - 1;
                //check if current patch is inside the image
                if(patch_start_pos_x >= 0 && patch_start_pos_y >= 0 && patch_end_pos_x < img_width && patch_end_pos_y < img_height){
                    patches_loc.push_back(w);
                    patches_loc.push_back(h);
                }
            }
        }
    }
    //ct.print_timer("Get valid pixels");
    int valid_patches = patches_loc.size() / 2;

    if(valid_patches > 0){
        //there are no valid patches
        if(valid_patches == 0){
            host_patches.resize(0);
            return;
        }

    //    dim3 threads_per_block(max_threads);
    //    dim3 num_blocks(valid_patches);


        int *dev_patches_loc;
        CUDA_SAFE_CALL( cudaMalloc(&dev_patches_loc, patches_loc.size() * sizeof(int)) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_patches_loc, &patches_loc[0], patches_loc.size()*sizeof(int), cudaMemcpyHostToDevice));

        float *dev_patches;
        int patches_size = valid_patches * patch_size_in_voxels * patch_size_in_voxels * 6;
        CUDA_SAFE_CALL( cudaMalloc(&dev_patches, patches_size * sizeof(float)) );
        //ct.print_timer("out malloc");

        extract<<<valid_patches, max_threads>>>(dev_patches, dev_patches_loc, stride_in_pixels, patch_size_in_voxels, voxel_size_in_m,
                                                focal_length, img_width, img_height, generate_random_values);
        //ct.print_timer("kernel exec");
        CUDA_CHECK_ERROR("after kernel");
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        host_patches.resize(patches_size);
        CUDA_SAFE_CALL( cudaMemcpy(&host_patches[0], dev_patches, patches_size * sizeof(float), cudaMemcpyDeviceToHost) );
        //ct.print_timer("copy output to host");
        CUDA_SAFE_CALL( cudaFree(dev_patches) );
        CUDA_SAFE_CALL( cudaFree(dev_patches_loc) );
    }

    CUDA_SAFE_CALL( cudaUnbindTexture(tex) );
    CUDA_SAFE_CALL( cudaFreeArray(dev_texture_arr) );
    //ct.print_timer("free memory");


}








extern "C"
__global__
void extract_rgbd(float *dev_patches, int *dev_patches_loc, int stride, int patch_size_in_voxels, float voxel_size_in_m, float max_depth_range_in_m,
                  float focal_length, int img_width, int img_height, bool generate_random_values){

    __shared__ float r_rand, g_rand, b_rand, d_rand;

    if(threadIdx.x == 0){
        unsigned int seed = static_cast<unsigned  int>( clock64() );
        curandState s;
        curand_init(seed, 0, 0, &s);
        r_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
        g_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
        b_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
        d_rand = static_cast<float>(curand(&s) % 255) / 255.0f;
    }
    __syncthreads();


    int patch_center_x = dev_patches_loc[blockIdx.x*2];
    int patch_center_y = dev_patches_loc[blockIdx.x*2 +1];

    //get depth of central pixel
    float patch_center_d = tex3D(tex, 3 + 0.5f, patch_center_x + 0.5f, patch_center_y + 0.5f)/1000.0f; //channel 4 (index 3) is Depth

    // get patch size according to current depth
    // validity of patch_center_d and patch lying inside image already checked
    int adaptive_patch_size_in_pixels = static_cast<int> ( static_cast<float>(patch_size_in_voxels) * voxel_size_in_m / patch_center_d * focal_length );
    int patch_start_pos_x = patch_center_x - adaptive_patch_size_in_pixels/2;
    int patch_start_pos_y = patch_center_y - adaptive_patch_size_in_pixels/2;

    //cyclic assignment of pixels to threads
    int cur_patch_pixel = threadIdx.x;
    while(cur_patch_pixel < patch_size_in_voxels * patch_size_in_voxels){

        //get patch coordinates for the pixel
        int thread_x = cur_patch_pixel % patch_size_in_voxels;
        int thread_y = cur_patch_pixel / patch_size_in_voxels;
        //get output pixel offset
        int dev_patches_pos = blockIdx.x * patch_size_in_voxels * patch_size_in_voxels * 4 + thread_y * patch_size_in_voxels * 4 + thread_x * 4;

        //get interpolation coordinates for the pixel
        float texture_pos_x = static_cast<float>(patch_start_pos_x) + static_cast<float>(thread_x) * ( static_cast<float>(adaptive_patch_size_in_pixels) / static_cast<float>(patch_size_in_voxels) );
        float texture_pos_y = static_cast<float>(patch_start_pos_y) + static_cast<float>(thread_y) * ( static_cast<float>(adaptive_patch_size_in_pixels) / static_cast<float>(patch_size_in_voxels) );

        // get depth to see if it belongs to object
        float d = tex3D(tex,  3 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f)/1000.0f;
        if(d > 0){

            dev_patches[dev_patches_pos + 0] = tex3D(tex,  0 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
            dev_patches[dev_patches_pos + 1] = tex3D(tex,  1 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);
            dev_patches[dev_patches_pos + 2] = tex3D(tex,  2 + 0.5f, texture_pos_x + 0.5f, texture_pos_y + 0.5f);

            float trunc_d = (d - patch_center_d)/max_depth_range_in_m + 0.5f;
            if(trunc_d > 1.0f)
                trunc_d = 1.0f;
            if(trunc_d < 0.0f)
                trunc_d = 0.0f;

            dev_patches[dev_patches_pos + 3] = trunc_d;


        } else {

            if(generate_random_values){
                dev_patches[dev_patches_pos + 0] = b_rand;
                dev_patches[dev_patches_pos + 1] = g_rand;
                dev_patches[dev_patches_pos + 2] = r_rand;
                dev_patches[dev_patches_pos + 3] = d_rand;
            } else {
                dev_patches[dev_patches_pos + 0] = 0;
                dev_patches[dev_patches_pos + 1] = 0;
                dev_patches[dev_patches_pos + 2] = 0;
                dev_patches[dev_patches_pos + 3] = 0;
            }
        }

        cur_patch_pixel += blockDim.x;
    }
}




//in channel 4 we want:
//0: d-max_depth_range_in_m/2, 0.5: d, 1:d+max_depth_range_in_m/2

//texture_3D: IMG_HEIGHT x IMG_WIDTH x 4 (RGBD)
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
                     float distance_threshold)
{
    cuda_timer ct;
    //ct.start_timer();

    cuda_initializer cuda_config;
    cuda_config.deviceInit();
    //ct.print_timer("Cuda Init");

    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;


    cudaArray *dev_texture_arr;

    //first dimenstion is the most quickly changing one
    cudaExtent extend = make_cudaExtent(4, img_width, img_height);

    CUDA_SAFE_CALL( cudaMalloc3DArray(&dev_texture_arr, &tex.channelDesc, extend) );

    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr((void*)&texture_3D[0], 4 * sizeof(float), 4, img_width);
    params.dstArray = dev_texture_arr;
    params.extent = extend;
    params.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&params) );

    CUDA_SAFE_CALL( cudaBindTextureToArray(tex, dev_texture_arr, tex.channelDesc) );
    //ct.print_timer("Texture binding");


    int max_threads = patch_size_in_voxels * patch_size_in_voxels < cuda_config.deviceProp.maxThreadsPerBlock ?
        patch_size_in_voxels * patch_size_in_voxels : cuda_config.deviceProp.maxThreadsPerBlock;

    //get valid patch locations - [x1, y1, x2, y2 .. ]

    srand(time(NULL));

    patches_loc.resize(0);
    for(int h=0; h<img_height; h += stride_in_pixels){
        for(int w=0; w<img_width; w += stride_in_pixels){
            int tex3d_depth_pos = h*img_width*4 + w*4 + 3;
            if(texture_3D[tex3d_depth_pos] != 0 && (float)texture_3D[tex3d_depth_pos]/1000.0f < distance_threshold){
                // not the fastest way to get random patches, just a workaround
                if(rand() % 100 <= percent * 100){
                    int adaptive_patch_size_in_pixels = static_cast<int> ( static_cast<float>(patch_size_in_voxels) * voxel_size_in_m / (texture_3D[tex3d_depth_pos]/1000.0f) * focal_length );
                    int patch_start_pos_x = w - adaptive_patch_size_in_pixels/2;
                    int patch_end_pos_x = patch_start_pos_x + adaptive_patch_size_in_pixels - 1;
                    int patch_start_pos_y = h - adaptive_patch_size_in_pixels/2;
                    int patch_end_pos_y = patch_start_pos_y + adaptive_patch_size_in_pixels - 1;
                    //check if current patch is inside the image
                    if(patch_start_pos_x >= 0 && patch_start_pos_y >= 0 && patch_end_pos_x < img_width && patch_end_pos_y < img_height){
                        patches_loc.push_back(w);
                        patches_loc.push_back(h);
                    }
                }
            }
        }
    }
    //ct.print_timer("Get valid pixels");
    int valid_patches = patches_loc.size() / 2;

    if(valid_patches > 0){
        //there are no valid patches
        if(valid_patches == 0){
            host_patches.resize(0);
            return;
        }

    //    dim3 threads_per_block(max_threads);
    //    dim3 num_blocks(valid_patches);


        int *dev_patches_loc;
        CUDA_SAFE_CALL( cudaMalloc(&dev_patches_loc, patches_loc.size() * sizeof(int)) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_patches_loc, &patches_loc[0], patches_loc.size()*sizeof(int), cudaMemcpyHostToDevice));

        float *dev_patches;
        int patches_size = valid_patches * patch_size_in_voxels * patch_size_in_voxels * 4;
        CUDA_SAFE_CALL( cudaMalloc(&dev_patches, patches_size * sizeof(float)) );
        //ct.print_timer("out malloc");

        extract_rgbd<<<valid_patches, max_threads>>>(dev_patches, dev_patches_loc, stride_in_pixels, patch_size_in_voxels, voxel_size_in_m,
                                                     max_depth_range_in_m, focal_length, img_width, img_height, generate_random_values);
        //ct.print_timer("kernel exec");
        CUDA_CHECK_ERROR("after kernel");
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        host_patches.resize(patches_size);
        CUDA_SAFE_CALL( cudaMemcpy(&host_patches[0], dev_patches, patches_size * sizeof(float), cudaMemcpyDeviceToHost) );
        //ct.print_timer("copy output to host");
        CUDA_SAFE_CALL( cudaFree(dev_patches) );
        CUDA_SAFE_CALL( cudaFree(dev_patches_loc) );
    }

    CUDA_SAFE_CALL( cudaUnbindTexture(tex) );
    CUDA_SAFE_CALL( cudaFreeArray(dev_texture_arr) );
    //ct.print_timer("free memory");


}


} // namespace patch_extractor_gpu
