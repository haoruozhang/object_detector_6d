/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Generate samples for training Hough Forest
 */
#include <string>
#include <caffe/caffe.hpp>

class train_patch_generator {

    struct Annotation{
        std::string annot_key;
        float yaw, pitch, roll, obj_x, obj_y, obj_z;
    };


    std::string caffe_model_definition_filename_;
    std::string caffe_model_weights_filename_;
    std::string input_lmdb_;
    std::string output_file_;
    int patch_size_;
    bool use_gpu_;
    int batch_size_;

public:

    train_patch_generator(){
        caffe_model_definition_filename_ = "";
        caffe_model_weights_filename_ = "";
        input_lmdb_ = "";
        output_file_ = "";
        patch_size_ = 0;
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        use_gpu_ = false;
        batch_size_ = 1;
    }

    void useGPU(int dev_id){
        caffe::Caffe::SetDevice(dev_id);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        use_gpu_ = true;
    }

    void useCPU(){
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        use_gpu_ = false;
    }

    void setCaffeModel(const std::string &definition_filename, const std::string &weights_filename){
        caffe_model_definition_filename_ = definition_filename;
        caffe_model_weights_filename_ = weights_filename;
    }

    void setInputLmdb(const std::string &flmdb){
        input_lmdb_ = flmdb;
    }

    void setPatchSize(int psize_in_voxels){
        patch_size_ = psize_in_voxels;
    }

    void setOutputFile(const std::string &fout){
        output_file_ = fout;
    }

    void setBatchSize(int b){
        batch_size_ = b;
    }

    void generate_train_patches();
    void generate_train_patches_pixeltests();
    void generate_train_patches_kmeans_centers();
    void generate_train_patches_kmeans_vectors();


};
