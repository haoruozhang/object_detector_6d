#include <cuda/surface_normals.h>
#include <cuda/cuda_utils.h>
#include <iostream>

namespace surface_normals_gpu {

texture<float, 2, cudaReadModeElementType> tex;

//blockDim represents the patch size

extern "C"
__global__
void generate(float *dev_normals, int img_width, int img_height, float focal_length){
    int array_pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    int img_pos_x = array_pos % img_width;
    int img_pos_y = array_pos / img_width;

    if(img_pos_x > 0 && img_pos_x < img_width-1 && img_pos_y > 0 && img_pos_y < img_height-1){
        float z       = tex2D(tex, img_pos_x     + 0.5f, img_pos_y     + 0.5f)/1000.0f;
        float z_left  = tex2D(tex, img_pos_x - 1 + 0.5f, img_pos_y     + 0.5f)/1000.0f;
        float z_right = tex2D(tex, img_pos_x + 1 + 0.5f, img_pos_y     + 0.5f)/1000.0f;
        float z_up    = tex2D(tex, img_pos_x     + 0.5f, img_pos_y - 1 + 0.5f)/1000.0f;
        float z_down  = tex2D(tex, img_pos_x     + 0.5f, img_pos_y + 1 + 0.5f)/1000.0f;
        if(z != 0 && z_left != 0 && z_right != 0 && z_up != 0 && z_down != 0){
            float x_left = ( (float)img_pos_x - 1 - (float)img_width/2.0f ) * z_left / focal_length;
            float x_right = ( (float)img_pos_x + 1 - (float)img_width/2.0f ) * z_right / focal_length;
            float x_up = ( (float)img_pos_x - (float)img_width/2.0f ) * z_up / focal_length;
            float x_down = ( (float)img_pos_x - (float)img_width/2.0f ) * z_down / focal_length;

            float y_left = ( (float)img_pos_y - (float)img_height/2.0f ) * z_left / focal_length;
            float y_right = ( (float)img_pos_y - (float)img_height/2.0f ) * z_right / focal_length;
            float y_up = ( (float)img_pos_y - 1 - (float)img_height/2.0f ) * z_up / focal_length;
            float y_down = ( (float)img_pos_y + 1 - (float)img_height/2.0f ) * z_down / focal_length;

            //calculate gradients
            float ax = (x_left - x_right)/2.0f;
            float ay = (y_left - y_right)/2.0f;
            float az = (z_left - z_right)/2.0f;

            float bx = (x_down - x_up)/2.0f;
            float by = (y_down - y_up)/2.0f;
            float bz = (z_down - z_up)/2.0f;

            //cross product
            float nx = -(ay*bz - az*by);
            float ny = -(az*bx - ax*bz);
            float nz = -(ax*by - ay*bx);

            float mag = sqrt(nx*nx + ny*ny + nz*nz);
            nx /= mag;
            ny /= mag;
            nz /= mag;

            int normals_pos = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
            dev_normals[normals_pos] = nx;
            dev_normals[normals_pos+1] = ny;
            dev_normals[normals_pos+2] = nz;
        }
        else{
            int normals_pos = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
            dev_normals[normals_pos] = 0;
            dev_normals[normals_pos+1] = 0;
            dev_normals[normals_pos+2] = 0;
        }
    }
    else if(img_pos_x >= 0 && img_pos_x < img_width && img_pos_y >= 0 && img_pos_y < img_height){
        int normals_pos = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
        dev_normals[normals_pos] = 0;
        dev_normals[normals_pos+1] = 0;
        dev_normals[normals_pos+2] = 0;
    }

}

void generate_normals(const std::vector<unsigned short> &depthmap, int img_width, int img_height, float focal_length, std::vector<float> &host_normals){


    cuda_timer ct;
    ct.start_timer();

    cuda_initializer cuda_config;
    cuda_config.deviceInit();
    //ct.print_timer("Cuda Init");

    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;


    cudaArray *dev_depth_arr;
    std::vector<float> depthmap_float(depthmap.begin(), depthmap.end());
    CUDA_SAFE_CALL( cudaMallocArray(&dev_depth_arr, &tex.channelDesc, img_width, img_height) );
    CUDA_SAFE_CALL( cudaMemcpyToArray(dev_depth_arr, 0, 0, &depthmap_float[0], img_width*img_height*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaBindTextureToArray(tex, dev_depth_arr) );

    //ct.print_timer("Texture binding");

    int num_threads = 64;
    int num_blocks = (int)( img_width*img_height/num_threads + 0.5f);


    float *dev_normals;
    int normals_size = img_width * img_height * 3; //X, Y, Z
    CUDA_SAFE_CALL( cudaMalloc(&dev_normals, normals_size * sizeof(float)) );
    //ct.print_timer("out malloc");

    generate<<<num_blocks, num_threads>>>(dev_normals, img_width, img_height, focal_length);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    //ct.print_timer("kernel exec");
    CUDA_CHECK_ERROR("after kernel");



    host_normals.resize(normals_size);
    CUDA_SAFE_CALL( cudaMemcpy(&host_normals[0], dev_normals, normals_size * sizeof(float), cudaMemcpyDeviceToHost) );
    //ct.print_timer("copy output to host");
    CUDA_SAFE_CALL( cudaFree(dev_normals) );
    CUDA_SAFE_CALL( cudaUnbindTexture(tex) );
    CUDA_SAFE_CALL( cudaFreeArray(dev_depth_arr) );
    //ct.print_timer("free memory");

}

} //namespace surface_normals_gpu
