/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * HFTest class for using a trained forest to detect the
 * objects
 */

#ifndef HFTEST_H_
#define HFTEST_H_

#include <HFBase.h>
#include <iostream>
#include <caffe/caffe.hpp>
#include <cv.h>
#include <boost/unordered_map.hpp>

#include <MeshUtils.h>
#include <detector_options.pb.h>

#include <google/protobuf/text_format.h>
using namespace google::protobuf;

class HFTest : public HFBase
{

    std::string caffe_model_definition_filename_;
    std::string caffe_model_weights_filename_;
    std::string forest_folder_;
    int stride_in_pixels_;    
    bool use_gpu_;
    int num_threads_;
    float max_depth_range_in_m_;
    int batch_size_caffe_;

    std::string mesh_folder_;

    //for hypotheses calculation
    int max_yaw_pitch_hypotheses_;
    int max_roll_hypotheses_;
    float min_location_score_ratio_;
    float min_yaw_pitch_drop_ratio_;

    int centers_blur_size_;
    int centers_maxsupression_wsize_;
    int pose_2d_blur_size_;
    int pose_2d_maxsupression_wsize_;

    float fx_, fy_, cx_, cy_;

    typedef boost::unordered_map<HoughPoint, std::vector<TreeNode*>, HpHash> VotesToNodeMap;
    typedef std::pair<float, cv::Point> MapHypothesis;
    struct HypothesisComparator {
        inline bool operator()(const MapHypothesis &h1, const MapHypothesis &h2){
            return h1.first > h2.first;
        }
    };

    typedef boost::unordered_map<int, TreeNode*> LeafMap;
    typedef boost::unordered_map<int, boost::unordered_map<int, float> > GroupTransitions;
    struct CameraTransitions{
        Eigen::Vector3f camera_transition;
        GroupTransitions group_transitions;
    };

    typedef std::vector<std::vector<std::vector<CameraTransitions> > > TreeTransitions;

    typedef boost::unordered_map<int, boost::unordered_map<int, bool> > SparseMap;


    void detect(const std::vector<float> &test_vec, int patch_x, int patch_y, unsigned short depth, std::vector<float> &res, std::vector<cv::Mat> &obj_center_hough_2d, std::vector<VotesToNodeMap> &center_leaf_map, const DetectorOptions::Options &detect_options);
    Eigen::Vector3f get_obj_center_vote_from_6dof(Eigen::VectorXf dof, int patch_x, int patch_y, unsigned short depth);
    cv::Mat disp_depth(const cv::Mat &depth);
    Eigen::Vector2i Point3DToImage(const Eigen::Vector3f &p);
    void non_max_suppression(Hough3DMap &hough_map, Hough3DMap &hough_map_out);
    void non_max_suppression(const cv::Mat &input, std::vector<MapHypothesis> &local_maxima, cv::Size2i w);    


    void get_leaf_map(TreeNode *n, LeafMap &leaf_map);
    void get_tree_transitions(TreeTransitions &tree_transitions, SparseMap &leaf_to_group, SparseMap &group_to_leaf);
    void detect_temp(const std::vector<float> &test_vec,
                             SparseMap &leaf_to_group,
                             SparseMap &group_to_leaf,
                             TreeTransitions &tree_transitions,
                             const Eigen::Vector3f &cam_pose,
                             boost::unordered_map<int, float> &predicted_leafs);
    float get_cam_dist(const Eigen::Vector3f &cam1, const Eigen::Vector3f &cam2);

    void check_results(std::vector<VotesToNodeMap> &center_leaf_map, MeshUtils &mesh_utils, const Eigen::MatrixXi &patch_classes);    
    void correctPose(Eigen::Matrix4f& objpose, Eigen::Vector4f coeffs);


protected:
    HFTest::TreeNode* get_leaf(TreeNode *node, const std::vector<float> &test_vec);

public:
    HFTest()
        : stride_in_pixels_(2)
        , use_gpu_(false)
        , fx_(575)
        , fy_(575)
        , cx_(319.5f)
        , cy_(239.5f)
        , num_threads_(1)
        , max_depth_range_in_m_(0.25f)
        , batch_size_caffe_(100)
    {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);

        //for hypotheses calculation
        setHypothesesCalculationOption();
    }

    std::vector<MeshUtils::ObjectHypothesis> test_image(cv::Mat& rgb,  cv::Mat& depth,
                 const DetectorOptions::Options &detect_options, MeshUtils &mesh_utils,
                 bool generate_random_values, float patch_distance_threshold);

    void DetectObjects();

    void setCaffeModel(const std::string &definition_filename, const std::string &weights_filename){
        caffe_model_definition_filename_ = definition_filename;
        caffe_model_weights_filename_ = weights_filename;
    }

    void setBatchSizeCaffe(int sz) {
        batch_size_caffe_ = sz;
    }

    bool setInputForest(const std::string &forest_folder){
        std::cout << "Loading forest in " << forest_folder << "..." << std::endl;
        forest_folder_ = forest_folder;
        loadForestFromFolder(forest_folder);
        if(is_forest_loaded_) {
            std::cout << "Forest loaded successfully." << std::endl;
            return true;
        }
        std::cout << "Could not load forest." << std::endl;
        return false;
    }

    void useGPU(bool gpu_dev){
        use_gpu_ = gpu_dev >= 0;
        if(use_gpu_) {
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
            caffe::Caffe::SetDevice(gpu_dev);
        }
        else {
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
        }
    }

    void setStrideInPixels(int stride){
        stride_in_pixels_ = stride;
    }

    void setCameraIntrinsics(float fx, float fy, float cx, float cy){
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }

    std::string getForestFolder(){
        return forest_folder_;
    }

    void setNumThreads(int threads){
        num_threads_ = threads;
    }

    void setMaxDepthRange(float d){
        max_depth_range_in_m_ = d;
    }

    void setHypothesesCalculationOption(
        int max_yaw_pitch_hypotheses = 7,
        int max_roll_hypotheses = 3,
        float min_location_score_ratio = 1.0f / 1000.0f,
        float min_yaw_pitch_drop_ratio = 1.0f / 1000.0f,
        int centers_blur_size = 13,
        int centers_maxsupression_wsize = 40,
        int pose_2d_blur_size = 35, /*degrees*/
        int pose_2d_maxsupression_wsize = 35) {

            max_yaw_pitch_hypotheses_ = max_yaw_pitch_hypotheses;
            max_roll_hypotheses_ = max_roll_hypotheses;
            min_location_score_ratio_ = min_location_score_ratio;
            min_yaw_pitch_drop_ratio_ = min_yaw_pitch_drop_ratio;
            centers_blur_size_ = centers_blur_size;
            centers_maxsupression_wsize_ = centers_maxsupression_wsize;
            pose_2d_blur_size_ = pose_2d_blur_size;
            pose_2d_maxsupression_wsize_ = pose_2d_maxsupression_wsize;

    }

    void setMeshFolder(std::string folder) {
        mesh_folder_ = folder;
    }

};

#endif
