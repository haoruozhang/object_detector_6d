/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Generating and saving patches from image
 */
#ifndef PATCH_GENERATOR_H
#define PATCH_GENERATOR_H

#include <string>
#include <vector>
#include <sys/time.h>
#include <math.h>
#include <cuda/patch_extractor.h>
#include <cv.h>
#include <my_c_timer.h>
#include <caffe/proto/caffe.pb.h>
#include <lmdb.h>

#include <Eigen/Dense>

class patch_generator
{

    std::vector<std::string> input_object_folders_;
    std::string output_folder_;
    int num_objects_;
    int stride_in_pixels_;
    float voxel_size_in_m_;
    int patch_size_in_voxels_;
    float rendering_view_angle_;
    bool generate_random_values_;
    float distance_threshold_;
    float max_depth_range_in_m_;
    float percent_;

    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;


    float getFocalLength(int im_width, int im_height){
        return (float)im_height / 2.0f / (float)tan(rendering_view_angle_/ 180.0f * 3.141592f / 2.0f);
    }

    void insert_patches_to_db(const std::vector<float> &patches,
                                               caffe::Datum &datum,
                                               int cur_obj, int &patch_id,
                                               const Eigen::Matrix4f &obj_pose,
                                               const cv::Mat &depth,
                                               const std::vector<int> patches_loc,
                                               std::ofstream &fannot);

    void insert_patches_to_db_rgbd(const std::vector<float> &patches,
                                               caffe::Datum &datum,
                                               int cur_obj, int &patch_id,
                                               const Eigen::Matrix4f &obj_pose,
                                               const cv::Mat &depth,
                                               const std::vector<int> patches_loc,
                                               std::ofstream &fannot);

    void get_yaw_pitch_roll_from_rot_mat(const Eigen::Matrix4f &rot_mat, float &yaw, float &pitch, float &roll);
    Eigen::Vector4f get_object_coords(int im_width, int im_height, int w, int h, unsigned short depth, const Eigen::Matrix4f &rot_mat);



public:

    enum output_type {
       OUTPUT_LMDB,
       OUTPUT_BIN
    } out_type_;

    patch_generator(){

        //Give default values
        rendering_view_angle_ = 45.3105;
        patch_size_in_voxels_ = 16;
        //1mm is approximately 1 pixel in 60cm distance from the camera
        voxel_size_in_m_ = 0.001;
        stride_in_pixels_ = 5;
        out_type_ = OUTPUT_LMDB;
        generate_random_values_ = true;
        distance_threshold_ = 3.0f;
        max_depth_range_in_m_ = 0.25f;
        percent_ = 1;
    }

    void setOutputType(output_type t){
        out_type_ = t;
    }

    void setInputObjectFolders(const std::vector<std::string> &inp_obj_fold){
        input_object_folders_ = inp_obj_fold;
        num_objects_ = input_object_folders_.size();
    }

    void setOutputFolder(std::string &out_obj_folder){
        output_folder_ = out_obj_folder;
    }

    void setStride(int stride){
        stride_in_pixels_= stride;
    }

    void setVoxelSizeInM(float vxsz){
        voxel_size_in_m_ = vxsz;
    }

    void setPatchSizeInVoxels(int ptchsz){
        patch_size_in_voxels_ = ptchsz;
    }   

    void setRenderingViewAngle(float rv){
        rendering_view_angle_ = rv;
    }

    void setGenerateRandomValues(bool b){
        generate_random_values_ = b;
    }

    void setDistanceThreshold(float thres){
        distance_threshold_ = thres;
    }

    void setMaxDepthRangeInM(float d){
        max_depth_range_in_m_ = d;
    }

    void setPercent(float p){
        percent_ = p;
    }

    void generatePatches();
    void generatePatches_rgbd();

};

#endif
