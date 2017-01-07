/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * MeshUtils class, helper functions for detection
 */
#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include <string>
#include <vector>
#include <fstream>

#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPLYReader.h>

#include <glog/logging.h>

class MeshUtils {      

    struct BoundingBox {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud;
        float max_center_length;        
    };

    void getPointCloudFromPLY(const std::string &filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_pc, BoundingBox &bb);

    boost::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr > object_pointclouds_;
    boost::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr > object_pointclouds_downsampled_;
    boost::unordered_map<int, BoundingBox> object_bounding_boxes_;
    boost::unordered_map<int, vtkSmartPointer<vtkPolyData> > object_polydata_;
    //ground truth poses are in vtk frame
    boost::unordered_map<int, std::vector<Eigen::Matrix4f> > object_groundtruth_poses_;
    boost::unordered_map<int, bool> object_symmetric_;
    boost::unordered_map<int, pcl::PointCloud<pcl::Normal>::Ptr> object_downsampled_normals_;
    boost::unordered_map<int, std::string> object_names_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud_downsampled_;
    pcl::PointCloud<pcl::Normal>::Ptr scene_downsampled_normals_;
    std::vector<pcl::PointIndices> scene_clusters_;
    typename pcl::search::KdTree<pcl::PointXYZRGB>::Ptr scene_downsampled_tree_;

    boost::unordered_map<int, int> scene_indices_to_cluster_;

    cv::Mat scene_rgb_;
    cv::Mat scene_depth_;
    float fx_, fy_, cx_, cy_;

    float similarity_reg_;
    float inliers_reg_;
    float clutter_reg_;
    float location_score_reg_;
    float pose_score_reg_;
    float total_explained_ratio_reg_;
    float common_explained_ratio_reg_;
    float nn_search_radius_;
    float occlusion_threshold_;
    float final_score_threshold_;
    float clutter_threshold_;
    float inliers_threshold_;    
    bool single_object_instance_;
    bool single_object_in_group_;
    bool use_color_similarity_;
    bool use_normal_similarity_;

    Eigen::Vector4f up_vector_;
    boost::unordered_set<int> obj_correct_up_vector_;
    boost::unordered_map<int, float> obj_nn_search_radius_;
    boost::unordered_map<int, float> obj_icp_iterations_;

    //downsampling
    float scene_ds_leaf_size_;
    float object_ds_leaf_size_;

    //scene clustering
    double eps_angle_threshold_;
    int min_points_;
    float curvature_threshold_;
    float tolerance_near_;
    float tolerance_far_;

    int num_threads_;    

    cv::Mat getYawMat(float yaw);
    cv::Mat getPitchMat(float pitch);
    cv::Mat getRollMat(float roll);

    Eigen::Matrix4f get_rotmat_from_yaw_pitch_roll(float yaw, float pitch, float roll);
    void world_to_image_coords(float x, float y, float z, int &row, int &col);


    void get_normals_not_nan(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);
    template<typename PointT, typename NormalT>
    inline void extractEuclideanClustersSmooth(const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance_near, float tolerance_far,
        const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle, float curvature_threshold,
        unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());    

    bool plane_detected_;


public:
    MeshUtils(){

        fx_ = 575;
        fy_ = 575;
        cx_ = 319.5;
        cy_ = 239.5;

        similarity_reg_ = 10;
        inliers_reg_ = 2;
        clutter_reg_ = 2;
        location_score_reg_ = 1.0f;
        pose_score_reg_ = 1.0f;

        total_explained_ratio_reg_ = 1.0f;
        common_explained_ratio_reg_ = 1.0f;

        nn_search_radius_ = 0.01f;
        occlusion_threshold_ = 0.02f;

        final_score_threshold_ = 10.0f;
        clutter_threshold_ = 0.5f;
        inliers_threshold_ = 0.5f;

        single_object_instance_ = false;
        single_object_in_group_ = false;

        use_color_similarity_ = true;
        use_normal_similarity_ = true;

        scene_ds_leaf_size_ = 0.005f;
        object_ds_leaf_size_ = 0.005f;

        eps_angle_threshold_ = 0.05;
        min_points_ = 5;
        curvature_threshold_ = 0.1;
        tolerance_near_ = 0.03f;
        tolerance_far_ = 0.05f;

        num_threads_ = 1;
        plane_detected_ = false;
    }

    struct HypothesisEvaluation {
        float clutter_score;
        float similarity_score;
        float inliers_ratio;
        float visibility_ratio;
        float location_score;
        float pose_score;
        float ground_truth_error;
        float final_score;

        boost::unordered_map<int, bool> scene_explained_indices;
        int total_scene_indices_explained;

        HypothesisEvaluation():clutter_score(0),
            similarity_score(0), inliers_ratio(0),
            visibility_ratio(0), total_scene_indices_explained(0), pose_score(0),
            location_score(0), ground_truth_error(0), final_score(0)
        {}

        void operator=(const HypothesisEvaluation &he){
            clutter_score = he.clutter_score;
            similarity_score = he.similarity_score;
            inliers_ratio = he.inliers_ratio;
            visibility_ratio = he.visibility_ratio;
            location_score = he.location_score;
            pose_score = he.pose_score;
            ground_truth_error = he.ground_truth_error;
            scene_explained_indices = he.scene_explained_indices;
            total_scene_indices_explained = he.total_scene_indices_explained;
            final_score = he.final_score;
        }

    };


    struct ObjectHypothesis{
        int obj_id;
        Eigen::Matrix4f rotmat;
        HypothesisEvaluation eval;

        ObjectHypothesis():obj_id(-1) {}

        ObjectHypothesis(int obj_id, const Eigen::Matrix4f &rotmat):obj_id(obj_id), rotmat(rotmat) {}

        void operator=(const ObjectHypothesis &oh){
            obj_id = oh.obj_id;
            rotmat = oh.rotmat;
            eval = oh.eval;
        }     

    };

    typedef boost::unordered_map<int, boost::unordered_map<int, bool> > MutualExclusiveMap;

    void setIntrinsics(float fx, float fy, float cx, float cy){
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }


    void setScene(const cv::Mat &rgb, const cv::Mat &depth, float distance_threshold=2.0f);

    void insertObjectFromPLY(const std::string &filename, int obj_id, std::string obj_name = "",
                             bool correct_up_vector = false, float nn_search_radius = -1.0,
                             int icp_iterations = -1)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_pc ( new pcl::PointCloud<pcl::PointXYZRGB>() );
        BoundingBox bb;        
        getPointCloudFromPLY(filename, obj_pc, bb);
        object_pointclouds_[obj_id] = obj_pc;        
        object_bounding_boxes_[obj_id] = bb;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_pc_downsampled ( new pcl::PointCloud<pcl::PointXYZRGB>() );
        pcl::VoxelGrid<pcl::PointXYZRGB> downsample;
        downsample.setInputCloud(obj_pc);
        downsample.setLeafSize(object_ds_leaf_size_, object_ds_leaf_size_, object_ds_leaf_size_);
        downsample.filter(*obj_pc_downsampled);

        pcl::PointCloud<pcl::Normal>::Ptr obj_normals ( new pcl::PointCloud<pcl::Normal> () );
        get_normals_not_nan(obj_pc_downsampled, obj_normals);

        object_pointclouds_downsampled_[obj_id] = obj_pc_downsampled;
        object_downsampled_normals_[obj_id] = obj_normals;

        //read PLY
        vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
        reader->SetFileName (filename.c_str());

        object_polydata_[obj_id] = reader->GetOutput();
        object_polydata_[obj_id]->Update();

        object_names_[obj_id] = obj_name;
        if(correct_up_vector)
            obj_correct_up_vector_.insert(obj_id);

        if(nn_search_radius != -1.0)
            obj_nn_search_radius_[obj_id] = nn_search_radius;

        if(icp_iterations != -1)
            obj_icp_iterations_[obj_id] = icp_iterations;        

    }

    //ground truth pose should be in vtk format
    void insertObjectGroundtruthPose(int obj_id, const Eigen::Matrix4f &p, bool is_symmetric=false){
        object_groundtruth_poses_[obj_id].push_back(p);
        object_symmetric_[obj_id] = is_symmetric;
    }

    //works for single object in pose file. For multiple use the above
    void insertObjectGroundtruthPose(int obj_id, const std::string &filename, bool is_symmetric=false){
        Eigen::Matrix4f p;
        std::ifstream f(filename.c_str());
        CHECK(f) << "Cannot open annotation file: " << filename;
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j)
                f >> p(i, j);

        insertObjectGroundtruthPose(obj_id, p, is_symmetric);
    }

    Eigen::Matrix4f getGroundTruthPose(int obj_id, int n){
        CHECK(object_groundtruth_poses_.count(obj_id)) << "Asking for groundtruth pose that is not available";
        CHECK_GT(object_groundtruth_poses_[obj_id].size(), n) << "Asking no. of pose that is not available";
        Eigen::Matrix4f corrMat;
        corrMat << 1,  0,  0, 0,
                   0, -1,  0, 0,
                   0,  0, -1, 0,
                   0,  0,  0, 1;

        return corrMat * object_groundtruth_poses_[obj_id][n];
    }

    cv::Mat getObjMask(int obj_id, Eigen::Matrix4f rotmat);


    void setNNSearchRadius(float nn_search_radius){
        nn_search_radius_ = nn_search_radius;
    }

    void setOcclusionThreshol(float occthres){
        occlusion_threshold_ = occthres;
    }

    void setFinalScoreThreshold(float thres){
        final_score_threshold_ = thres;
    }

    void setClutterThreshold(float thres){
        clutter_threshold_ = thres;
    }

    void setInliersThreshold(float thres){
        inliers_threshold_ = thres;
    }

    void setReg(float similarity_reg, float inliers_reg, float clutter_reg, float location_score_reg, float pose_score_reg ){
        similarity_reg_ = similarity_reg;
        inliers_reg_ = inliers_reg;
        clutter_reg_ = clutter_reg;
        pose_score_reg_ = pose_score_reg;
        location_score_reg_ = location_score_reg;
    }

    void setGroupReg(float total_explained_ratio_reg, float common_explained_ratio_reg){
        total_explained_ratio_reg_ = total_explained_ratio_reg;
        common_explained_ratio_reg_ = common_explained_ratio_reg;
    }

    void searchSingleObjectInstance(bool single){
        single_object_instance_ = single;
    }

    //each separate group of hypotheses, should contain only one instance
    //for objects far apart
    //be carefull if a group contains more than one good gypotheses,
    //it will take only one
    void searchSingleObjectInGroup(bool single_group){
        single_object_in_group_ = single_group;
    }

    void useColorSimilarity(bool b){
        use_color_similarity_ = b;
    }

    void useNormalSimilarity(bool b){
        use_normal_similarity_ = b;
    }

    void setNumThreads(int n){
        num_threads_ = n;
    }

    void setDownsamplingSizes(float scene_ds_leaf_size = 0.005f, float object_ds_leaf_size = 0.005f) {
        scene_ds_leaf_size_ = scene_ds_leaf_size;
        object_ds_leaf_size_ = object_ds_leaf_size;
    }

    void setClusteringOptions(float eps_angle_threshold = 0.05,
        int min_points = 5,
        float curvature_threshold = 0.1,
        float tolerance_near = 0.03f,
        float tolerance_far = 0.05f)
    {

            eps_angle_threshold_ = eps_angle_threshold;
            min_points_ = min_points;
            curvature_threshold_ = curvature_threshold;
            tolerance_near_ = tolerance_near;
            tolerance_far_ = tolerance_far;
    }

    Eigen::Vector4f getUpVector() {
        return up_vector_;
    }

    bool is_plane_detected() {
        return plane_detected_;
    }


    void renderObject(cv::Mat& rgb_out, int obj_id, int row, int col, float z, float yaw, float pitch, float roll, float alpha);
    void renderObject(cv::Mat& rgb_out, int obj_id, float x, float y, float z, float yaw, float pitch, float roll, float alpha);
    void renderObject(cv::Mat& rgb_out, int obj_id, const Eigen::Matrix4f &rotmat, float alpha);    
    void icp(int obj_id, int row, int col, float z, float yaw, float pitch, float roll, Eigen::Matrix4f &rotmat, int iter=50);
    bool evaluate_hypothesis(ObjectHypothesis &h, float location_hough_score, float pose_score);
    void draw_hypotheses_boundingbox(const ObjectHypothesis &h, cv::Mat &rgb, const cv::Scalar &color, int line_width=1);
    std::vector<int> optimize_hypotheses(std::vector<ObjectHypothesis> &hypotheses);
    bool checkHypothesisWithGroundtruth(ObjectHypothesis &h);


private:

    bool get_next_solution_vector(std::vector<bool> &solution, const std::vector<int> &hypotheses_group,
                                  MutualExclusiveMap &mutual_exclusive_hypotheses);

    std::vector<int> optimize_hypotheses_single(std::vector<ObjectHypothesis> &hypotheses);
    std::vector<int> optimize_hypotheses_multi(std::vector<ObjectHypothesis> &hypotheses);
    void correctPose(Eigen::Matrix4f& objpose);

};

#endif
