#include <fstream>

#include <MeshUtils.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <vtkTransformFilter.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTransform.h>
#include <vtkMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGridGeometryFilter.h>
#include <vtkFloatArray.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>


//when rotmat is used, it is assumed to be in the xtion coordinate frame
//when yaw, pitch, roll is used, it is assumed they come from forest leaves
//that contain angles in vtk camera frame.

Eigen::Matrix4f MeshUtils::get_rotmat_from_yaw_pitch_roll(float yaw, float pitch, float roll){

    //get camera prediction -> xtion_rotmat
    float cos_yaw = cos(yaw);
    float sin_yaw = sin(yaw);
    float cos_pitch = cos(pitch);
    float sin_pitch = sin(pitch);
    float cos_roll = cos(roll);
    float sin_roll = sin(roll);

    Eigen::Matrix4f Rz;
    Rz << cos_yaw, -sin_yaw,  0, 0,
          sin_yaw,  cos_yaw,  0, 0,
          0,        0,        1, 0,
          0,        0,        0, 1;

    Eigen::Matrix4f Ry;
    Ry <<  cos_pitch,  0,  sin_pitch, 0,
          0,           1,  0,         0,
          -sin_pitch,  0,  cos_pitch, 0,
          0,           0,  0,         1;

    Eigen::Matrix4f Rx;
    Rx << 1,  0,          0,        0,
          0,  cos_roll,  -sin_roll, 0,
          0,  sin_roll,   cos_roll, 0,
          0,  0,          0,        1;

    return (Rz * Ry * Rx);

}

void MeshUtils::world_to_image_coords(float x, float y, float z, int &row, int &col){
    row = int(y * fy_ / z + cy_);
    col = int(x * fx_ / z + cx_);
}


void MeshUtils::getPointCloudFromPLY(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_pc, BoundingBox &bb){

    std::ifstream f(filename.c_str());
    CHECK(f) << "Cannot open file " << filename;

    std::string inp;
    int nVertex;
    while(!f.eof()){
        f >> inp;
        if(inp.compare("element")==0){
            f >> inp;
            if(inp.compare("vertex")==0)
                f >> nVertex;
        }
        if(inp.compare("end_header")==0)
            break;
    }
    obj_pc->points.resize(0);


    float min_x = FLT_MAX;
    float min_y = FLT_MAX;
    float min_z = FLT_MAX;
    float max_x = -FLT_MAX;
    float max_y = -FLT_MAX;
    float max_z = -FLT_MAX;
    float mean_x = 0;
    float mean_y = 0;
    float mean_z = 0;

    float x,y,z;
    int r,g,b,a;
    for(int i=0; i<nVertex; ++i){
        f >> x >> y >> z >> r >> g >> b >> a;        
        pcl::PointXYZRGB p(r,g,b);
        p.x = x; p.y = y; p.z = z;
        obj_pc->points.push_back(p);
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
        mean_x += x / (float)nVertex;
        mean_y += y / (float)nVertex;
        mean_z += z / (float)nVertex;
    }   

    obj_pc->width = obj_pc->points.size();
    obj_pc->height = 1;

    bb.pointcloud.reset( new pcl::PointCloud<pcl::PointXYZ>() );
    bb.pointcloud->width = 8;
    bb.pointcloud->height = 1;
    bb.pointcloud->points.push_back( pcl::PointXYZ(min_x, min_y, min_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(max_x, min_y, min_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(min_x, max_y, min_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(max_x, max_y, min_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(min_x, min_y, max_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(max_x, min_y, max_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(min_x, max_y, max_z) );
    bb.pointcloud->points.push_back( pcl::PointXYZ(max_x, max_y, max_z) );

    bb.max_center_length = 0;
    for(int i=0; i<8; ++i){
        pcl::PointXYZ *p = &(bb.pointcloud->points[i]);
        float dist = sqrt( pow(mean_x - p->x, 2.0f) + pow(mean_y - p->y, 2.0f) + pow(mean_z - p->z, 2.0f) );
        bb.max_center_length = std::max(bb.max_center_length, dist);
    }

}


//align the z axis of an object hypotheses with the normal of the table
//coeffs contains the vertical vector pointing -above- the table
//(given by pcl segmentation
void MeshUtils::correctPose(Eigen::Matrix4f& objpose){

    Eigen::Vector3f n;
    n << up_vector_(0), up_vector_(1), up_vector_(2);
    n.normalize();

    Eigen::Vector3f x_proj_on_plane;
    //inner product x.n
    float val = 0;
    for(int i=0; i<3; ++i)
        val += objpose(i,0) * n(i);
    for(int i=0; i<3; ++i){
        x_proj_on_plane(i) = objpose(i,0) - val*n(i);
    }
    x_proj_on_plane.normalize();
    int temp = x_proj_on_plane.dot(n);
    Eigen::Vector3f y = -x_proj_on_plane.cross(n);
    for(int i=0; i<3; ++i)
        objpose(i, 0) = x_proj_on_plane(i);
    for(int i=0; i<3; ++i)
        objpose(i, 1) = y(i);
    for(int i=0; i<3; ++i)
        objpose(i, 2) = n(i);

}



void MeshUtils::draw_hypotheses_boundingbox(const ObjectHypothesis &h, cv::Mat &rgb, const cv::Scalar &color, int line_width){

    pcl::PointCloud<pcl::PointXYZ> bb_trans;
    pcl::transformPointCloud(*(object_bounding_boxes_[h.obj_id].pointcloud), bb_trans, h.rotmat);
    std::vector<cv::Point> bb_points(8);
    for(int i=0; i<8; ++i)
        world_to_image_coords(bb_trans.points[i].x, bb_trans.points[i].y, bb_trans.points[i].z, bb_points[i].y, bb_points[i].x);   

    cv::line(rgb, bb_points[0], bb_points[1], color, line_width);
    cv::line(rgb, bb_points[0], bb_points[2], color, line_width);
    cv::line(rgb, bb_points[1], bb_points[3], color, line_width);
    cv::line(rgb, bb_points[2], bb_points[3], color, line_width);
    cv::line(rgb, bb_points[4], bb_points[5], color, line_width);
    cv::line(rgb, bb_points[4], bb_points[6], color, line_width);
    cv::line(rgb, bb_points[5], bb_points[7], color, line_width);
    cv::line(rgb, bb_points[6], bb_points[7], color, line_width);
    cv::line(rgb, bb_points[0], bb_points[4], color, line_width);
    cv::line(rgb, bb_points[1], bb_points[5], color, line_width);
    cv::line(rgb, bb_points[2], bb_points[6], color, line_width);
    cv::line(rgb, bb_points[3], bb_points[7], color, line_width);

}

//computes normals of point cloud
//changes also point_cloud if there are nan normals
void MeshUtils::get_normals_not_nan(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals) {

    float radius_normals = 0.03f;

    typename pcl::search::KdTree<pcl::PointXYZRGB>::Ptr normals_tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    typedef typename pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> NormalEstimator_;
    NormalEstimator_ n3d;

    normals_tree->setInputCloud (point_cloud);

    n3d.setRadiusSearch (radius_normals);
    n3d.setSearchMethod (normals_tree);
    n3d.setInputCloud (point_cloud);
    n3d.compute (*normals);

    //check nans...
    int j = 0;
    for (size_t i = 0; i < normals->points.size (); ++i)
    {
        if (!pcl_isfinite (normals->points[i].normal_x) || !pcl_isfinite (normals->points[i].normal_y)
            || !pcl_isfinite (normals->points[i].normal_z))
          continue;

        normals->points[j] = normals->points[i];
        point_cloud->points[j] = point_cloud->points[i];

        j++;
    }

    normals->points.resize (j);
    normals->width = j;
    normals->height = 1;

    point_cloud->points.resize (j);
    point_cloud->width = j;
    point_cloud->height = 1;

}



///////Function in Global Verification//////////

////TODO
/// Some points do not belong to a cluster sometimes!
///
///

template<typename PointT, typename NormalT>
inline void MeshUtils::extractEuclideanClustersSmooth(const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance_near, float tolerance_far,
    const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle, float curvature_threshold,
    unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster )
{

  if (tree->getInputCloud ()->points.size () != cloud.points.size ())
  {
    PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
    return;
  }
  if (cloud.points.size () != normals.points.size ())
  {
    PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
    return;
  }

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (cloud.points.size (), false);

  std::vector<int> nn_indices;
  std::vector<float> nn_distances;
  // Process all points in the indices vector
  int size = static_cast<int> (cloud.points.size ());
  for (int i = 0; i < size; ++i)
  {
    if (processed[i])
      continue;

    std::vector<unsigned int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back (i);

    processed[i] = true;

    while (sq_idx < static_cast<int> (seed_queue.size ()))
    {

      if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
      {
        sq_idx++;
        continue;
      }

      // Search for sq_idx
      float tolerance = cloud.points[seed_queue[sq_idx]].z > 1.3f ? tolerance_far : tolerance_near;
      if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
      {
        sq_idx++;
        continue;
      }

      for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
      {
        if (processed[nn_indices[j]]) // Has this point been processed before ?
          continue;

        if (normals.points[nn_indices[j]].curvature > curvature_threshold)
        {
          continue;
        }

        double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
            + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1]
            + normals.points[seed_queue[sq_idx]].normal[2] * normals.points[nn_indices[j]].normal[2];

        if (fabs (acos (dot_p)) < eps_angle)
        {
          processed[nn_indices[j]] = true;
          seed_queue.push_back (nn_indices[j]);
        }
      }

      sq_idx++;
    }

    // If this queue is satisfactory, add to the clusters
    if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
    {
      pcl::PointIndices r;
      r.indices.resize (seed_queue.size ());
      for (size_t j = 0; j < seed_queue.size (); ++j){
        r.indices[j] = seed_queue[j];        
        scene_indices_to_cluster_[r.indices[j]] = clusters.size();
      }

      std::sort (r.indices.begin (), r.indices.end ());
      r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());

      r.header = cloud.header;
      clusters.push_back (r);
    }
  }
}

DECLARE_bool(show_scene);
void MeshUtils::setScene(const cv::Mat &rgb, const cv::Mat &depth, float distance_threshold){


    //create point cloud    
    scene_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    scene_cloud_->height = rgb.rows;
    scene_cloud_->width  = rgb.cols;
    scene_cloud_->points.resize(rgb.rows * rgb.cols);
    distance_threshold *= 1000;
    for(int row=0; row<rgb.rows; ++row){
        for(int col=0; col<rgb.cols; ++col){

            if(depth.at<unsigned short>(row, col) != 0 &&
               depth.at<unsigned short>(row, col) < distance_threshold){

                float z = (float)depth.at<unsigned short>(row, col) / 1000.0f;
                float x = (col - cx_) * z / fx_;
                float y = (row - cy_) * z / fy_;
                pcl::PointXYZRGB p(rgb.at<cv::Vec3b>(row,col)[2],
                                   rgb.at<cv::Vec3b>(row,col)[1], rgb.at<cv::Vec3b>(row,col)[0]);
                p.x = x; p.y = y; p.z = z;
                scene_cloud_->at(col, row) = p;                
            }
        }
    }

    //set images
    scene_rgb_ = rgb;
    scene_depth_ = depth;

    pcl::VoxelGrid<pcl::PointXYZRGB> downsample;
    downsample.setInputCloud(scene_cloud_);
    downsample.setLeafSize(scene_ds_leaf_size_, scene_ds_leaf_size_, scene_ds_leaf_size_);
    scene_cloud_downsampled_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr
            ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    downsample.filter(*scene_cloud_downsampled_);


    //extract smooth regions

    //get scene normals
    scene_downsampled_normals_.reset (new pcl::PointCloud<pcl::Normal> ());
    get_normals_not_nan(scene_cloud_downsampled_, scene_downsampled_normals_);

    //initialize kdtree for search
    scene_downsampled_tree_.reset (new pcl::search::KdTree<pcl::PointXYZRGB>);
    scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

    extractEuclideanClustersSmooth<pcl::PointXYZRGB, pcl::Normal>
            (*scene_cloud_downsampled_, *scene_downsampled_normals_,
             tolerance_near_, tolerance_far_,
             scene_downsampled_tree_, scene_clusters_,
             eps_angle_threshold_, curvature_threshold_, min_points_);


    //visualize clusters

    if (FLAGS_show_scene) {
        srand(655);
        for(int i=0; i<scene_clusters_.size(); ++i){
            pcl::PointIndices ind = scene_clusters_[i];
            cv::Scalar color_sc(rand() % 256, rand() % 256, rand() % 256);
            for(int j=0; j<ind.indices.size(); ++j){

                float x = scene_cloud_downsampled_->points[ind.indices[j]].x;
                float y = scene_cloud_downsampled_->points[ind.indices[j]].y;
                float z = scene_cloud_downsampled_->points[ind.indices[j]].z;
                int row = int(y * fy_ / z + cy_);
                int col = int(x * fx_ / z + cx_);
                //scene_rgb_.at<cv::Vec3b>(row, col) = color;
                cv::circle(scene_rgb_, cv::Point(col, row), 4, color_sc, 4);

            }
        }
        while(cv::waitKey(30)==-1)
            cv::imshow("scene", scene_rgb_);
    }

}


void MeshUtils::icp(int obj_id, int row, int col, float z, float yaw, float pitch, float roll,
                    Eigen::Matrix4f &rotmat, int iter){

    float x = ((float)col-cx_)*z / fx_;
    float y = ((float)row-cy_)*z / fy_;

    rotmat = get_rotmat_from_yaw_pitch_roll(yaw, pitch, roll);
    //rotate rotmat (vtk camera in training) 180 degrees around X
    //to match xtion coordinate system
    Eigen::Matrix4f corrMat;
    corrMat << 1,  0,  0, 0,
               0, -1,  0, 0,
               0,  0, -1, 0,
               0,  0,  0, 1;
    rotmat = corrMat * rotmat;
    rotmat(0, 3) = x;
    rotmat(1, 3) = y;
    rotmat(2, 3) = z;

    //icp and correct rotmat
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_trans_downsample( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::transformPointCloud(*(object_pointclouds_downsampled_[obj_id]), *obj_trans_downsample, rotmat);

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(obj_trans_downsample);
    icp.setInputTarget(scene_cloud_downsampled_);

    int iterations = obj_icp_iterations_.count(obj_id) ? obj_icp_iterations_[obj_id] : iter;

    icp.setMaximumIterations(iterations);
    //icp.setMaxCorrespondenceDistance(icp_max_distance_);
    icp.setMaxCorrespondenceDistance(obj_nn_search_radius_[obj_id]);
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    icp.align(aligned);

    if(icp.hasConverged())
        rotmat = icp.getFinalTransformation() * rotmat;

    if(obj_correct_up_vector_.count(obj_id))
        correctPose(rotmat);

}


void MeshUtils::renderObject(cv::Mat& rgb_out, int obj_id, int row, int col,
                             float z, float yaw, float pitch, float roll, float alpha){

    //float z = (float)depth.at<unsigned short>(row, col)/1000.0f;
    float x = ((float)col-cx_)*z / fx_;
    float y = ((float)row-cy_)*z / fy_;

    renderObject(rgb_out, obj_id, x, y, z, yaw, pitch, roll, alpha);

}


void MeshUtils::renderObject(cv::Mat& rgb_out, int obj_id, float x, float y, float z, float yaw, float pitch, float roll, float alpha)
{   

    Eigen::Matrix4f rotmat = get_rotmat_from_yaw_pitch_roll(yaw, pitch, roll);
    //rotate rotmat (vtk camera) 180 degrees around X
    //to match xtion coordinate system
    Eigen::Matrix4f corrMat;
    corrMat << 1,  0,  0, 0,
               0, -1,  0, 0,
               0,  0, -1, 0,
               0,  0,  0, 1;
    rotmat = corrMat * rotmat;
    rotmat(0, 3) = x;
    rotmat(1, 3) = y;
    rotmat(2, 3) = z;

    renderObject(rgb_out, obj_id, rotmat, alpha);


}


void MeshUtils::renderObject(cv::Mat& rgb_out, int obj_id, const Eigen::Matrix4f &rotmat, float alpha){

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_trans( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::transformPointCloud(*(object_pointclouds_[obj_id]), *obj_trans, rotmat);    

    cv::Mat depthFromModel = cv::Mat::zeros(scene_depth_.rows, scene_depth_.cols, CV_32FC1);

    std::pair<int, int> text_pos = std::make_pair(rgb_out.rows, rgb_out.cols);
    for(int p=0; p<object_pointclouds_[obj_id]->points.size(); ++p){

        pcl::PointXYZRGB *point = &(obj_trans->points[p]);

        int row = int(point->y * fy_ / point->z + cy_);
        int col = int(point->x * fx_ / point->z + cx_);
        if(text_pos > std::make_pair(row, col)) text_pos = std::make_pair(row, col);
        if(row >= 0 && row < scene_rgb_.rows && col >= 0 && col < scene_rgb_.cols){
            if(point->z < (float)scene_depth_.at<unsigned short>(row,col)/1000.0f ||
                    scene_depth_.at<unsigned short>(row,col) == 0){
                if(depthFromModel.at<float>(row,col) == 0 || depthFromModel.at<float>(row,col) > point->z){
                    depthFromModel.at<float>(row,col) = point->z;                    
                    float g = alpha + (1-alpha)*(float)rgb_out.at<cv::Vec3b>(row,col)[1]/255.0f;
                    rgb_out.at<cv::Vec3b>(row,col)[1] = (unsigned char) std::min((g*255.0f) + g, 255.0f);
                }
            }
        }
    }

    if(object_names_.count(obj_id)) {
        cv::putText(rgb_out, object_names_[obj_id], cv::Point(text_pos.second, text_pos.first),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

}



cv::Mat MeshUtils::getObjMask(int obj_id, Eigen::Matrix4f rotmat){

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_trans( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::transformPointCloud(*(object_pointclouds_[obj_id]), *obj_trans, rotmat);

    cv::Mat mask(scene_rgb_.rows, scene_rgb_.cols, CV_8UC1, cv::Scalar(0));

    for(int p=0; p<object_pointclouds_[obj_id]->points.size(); ++p){

        pcl::PointXYZRGB *point = &(obj_trans->points[p]);

        int row = int(point->y * fy_ / point->z + cy_);
        int col = int(point->x * fx_ / point->z + cx_);
        if(row >= 0 && row < scene_rgb_.rows && col >= 0 && col < scene_rgb_.cols)
            if( (float)scene_depth_.at<unsigned short>(row,col) == 0 || std::fabs(point->z - (float)scene_depth_.at<unsigned short>(row,col)/1000.0f) < 0.07)
                mask.at<uchar>(row, col) = 1;

    }

    return mask;

}



bool MeshUtils::checkHypothesisWithGroundtruth(ObjectHypothesis &h){

    if(!object_groundtruth_poses_.count(h.obj_id)){
        std::cout << "No ground truth provided of object id: " << h.obj_id << std::endl;
        return false;
    }

    float k = 0.3f;
    float diameter = object_bounding_boxes_[h.obj_id].max_center_length * 2.0f;

    //rotate rotmat (vtk camera) 180 degrees around X
    //to match xtion coordinate system
    Eigen::Matrix4f corrMat;
    corrMat << 1,  0,  0, 0,
               0, -1,  0, 0,
               0,  0, -1, 0,
               0,  0,  0, 1;

    //search all possible ground truth poses, in case of multiple instance
    //appearing on the scene
    float min_error = FLT_MAX;
    for(int p=0; p<object_groundtruth_poses_[h.obj_id].size(); ++p){

        Eigen::Matrix4f gt_rotmat = corrMat * object_groundtruth_poses_[h.obj_id][p];
        float dist = 0;
        for(int i=0; i<object_pointclouds_downsampled_[h.obj_id]->points.size(); ++i){

            pcl::PointXYZRGB *p = &(object_pointclouds_downsampled_[h.obj_id]->points[i]);
            Eigen::Vector4f v(p->x, p->y, p->z, 1);
            Eigen::Vector4f p1 = gt_rotmat * v;
            Eigen::Vector4f p2;
            if(!object_symmetric_[h.obj_id])
                p2 = h.rotmat  * v;
            else {
                float min_dist = FLT_MAX;
                for(int j=0; j<object_pointclouds_downsampled_[h.obj_id]->points.size(); ++j){
                    pcl::PointXYZRGB *p = &(object_pointclouds_downsampled_[h.obj_id]->points[j]);
                    Eigen::Vector4f v(p->x, p->y, p->z, 1);
                    Eigen::Vector4f p2_tmp = h.rotmat * v;
                    float dist_tmp = sqrt( pow(p1(0) - p2_tmp(0),2) + pow(p1(1) - p2_tmp(1),2) + pow(p1(2) - p2_tmp(2),2) );
                    if(min_dist > dist_tmp){
                        min_dist = dist_tmp;
                        p2 = p2_tmp;
                    }
                }

            }
            dist += sqrt( pow(p1(0) - p2(0),2) + pow(p1(1) - p2(1),2) + pow(p1(2) - p2(2),2) );
        }
        dist /= (float)object_pointclouds_downsampled_[h.obj_id]->points.size();
        float error = dist/diameter;
        if(error < min_error)
            min_error = error;

      }

    h.eval.ground_truth_error = min_error;

    if(h.eval.ground_truth_error < k)
        return true;
    else
        return false;

}


//evaluates hypothesis h, the results are written to the eval member variable of h.
bool MeshUtils::evaluate_hypothesis(ObjectHypothesis &h, float location_hough_score, float pose_score)
{

    if(h.rotmat(2,3) > 1.5f)
        return false;

    //transform object
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_trans( new pcl::PointCloud<pcl::PointXYZRGB>() );    
    pcl::transformPointCloud(*(object_pointclouds_downsampled_[h.obj_id]), *obj_trans, h.rotmat);
    pcl::PointCloud<pcl::Normal>::Ptr obj_normals_trans( new pcl::PointCloud<pcl::Normal>() );
    get_normals_not_nan(obj_trans, obj_normals_trans);


    //get visible points;
    pcl::PointIndices visible_points;
    for(int p=0; p<obj_trans->points.size(); ++p){
        int row,col;
        pcl::PointXYZRGB *cur_point = &(obj_trans->points[p]);
        world_to_image_coords(cur_point->x, cur_point->y, cur_point->z, row, col);
        //int wsize;
        float scene_d = float(scene_depth_.at<unsigned short>(row, col)) / 1000.0f;
        if(scene_d == 0 || cur_point->z < scene_d + occlusion_threshold_)
            visible_points.indices.push_back(p);
    }

    int inliers = 0;
    int not_in_cluster = 0;
    h.eval.similarity_score = 0;
    h.eval.total_scene_indices_explained = 0;
    std::vector<int> scene_clusters_explained_pixels(scene_clusters_.size(), 0);
    std::vector<int> model_clusters_explained_pixels(scene_clusters_.size(), 0);
    std::vector<bool> scene_indices_visited(scene_cloud_downsampled_->points.size(), false);

    float search_radius;
    if(obj_nn_search_radius_.count(h.obj_id))
        search_radius = obj_nn_search_radius_[h.obj_id];
    else
        search_radius = nn_search_radius_;

    for(int p=0; p<visible_points.indices.size(); ++p){
        pcl::PointXYZRGB *cur_point = &(obj_trans->points[visible_points.indices[p]]);
        pcl::Normal *cur_obj_normal = &(obj_normals_trans->points[visible_points.indices[p]]);

        //find the nearest neighbor,
        //i.e. the point with the lowest cost
        //in max_d range

        std::vector<int> nn_indices;
        std::vector<float> nn_distances;       
        //search in radius
        if( scene_downsampled_tree_->radiusSearch(*cur_point, search_radius, nn_indices, nn_distances, std::numeric_limits<int>::max ()) ) {

            float best_neighbor_score = 0;
            int neighbor_id = -1;
            int nn_count = nn_indices.size();
            //average cost of the nearest points
            cv::Mat temp;
            scene_rgb_.copyTo(temp);
            for(int nn=0; nn<nn_count; ++nn){

                //depth cost
                float depth_score = 1.0f - nn_distances[nn] / nn_search_radius_;
                //normal cost
                pcl::Normal *cur_scene_normal = &(scene_downsampled_normals_->points[nn_indices[nn]]);
                float normal_score = cur_scene_normal->normal_x * cur_obj_normal->normal_x +
                                    cur_scene_normal->normal_y * cur_obj_normal->normal_y +
                                    cur_scene_normal->normal_z * cur_obj_normal->normal_z;
                //make [-1,1] to [0,1]
                normal_score = normal_score/2.0f + 0.5f;                

                // other options are possible
                if(!use_normal_similarity_)
                    normal_score = 1;

                //color cost
                pcl::PointXYZRGB *cur_scene_point = &(scene_cloud_downsampled_->points[nn_indices[nn]]);                
                float color_score = std::max(fabs((float)cur_scene_point->r - (float)cur_point->r),
                                             fabs((float)cur_scene_point->g - (float)cur_point->g));
                color_score = 1.0f - std::max((double)color_score,
                                              fabs((float)cur_scene_point->b - (float)cur_point->b) ) / 255.0f;

//                cv::Mat scene_color(1, 1, CV_8UC3);
//                cv::Mat obj_color(1, 1, CV_8UC3);
//                scene_color.at<cv::Vec3b>(0) = cv::Vec3b(cur_scene_point->b, cur_scene_point->g, cur_scene_point->r);
//                obj_color.at<cv::Vec3b>(0) = cv::Vec3b(cur_point->b, cur_point->g, cur_point->r);
//                cv::cvtColor(scene_color, scene_color, CV_BGR2Lab);
//                cv::cvtColor(obj_color, obj_color, CV_BGR2Lab);
//                float color_cost = std::max( fabs( (float)scene_color.at<cv::Vec3b>(0)[1] - (float)obj_color.at<cv::Vec3b>(0)[1]),
//                                             fabs( (float)scene_color.at<cv::Vec3b>(0)[2] - (float)obj_color.at<cv::Vec3b>(0)[2])) / 255.0f;

                float neighbor_score;
                if(use_color_similarity_)
                    neighbor_score = (normal_score + depth_score + color_score)/3.0f;
                else
                    neighbor_score = (normal_score + depth_score) / 2.0f;



                if(neighbor_score > best_neighbor_score){
                    best_neighbor_score = neighbor_score;
                    neighbor_id = nn_indices[nn];
                }
                if(!scene_indices_visited[nn_indices[nn]]){                                        
                    if(scene_indices_to_cluster_.count(nn_indices[nn]))
                        scene_clusters_explained_pixels[scene_indices_to_cluster_[nn_indices[nn]]]++;

                    scene_indices_visited[nn_indices[nn]] = true;
                    h.eval.total_scene_indices_explained++;
                    h.eval.scene_explained_indices[nn_indices[nn]] = true;
                }

            }


            h.eval.similarity_score += best_neighbor_score;


            if(scene_indices_to_cluster_.count(neighbor_id))
                model_clusters_explained_pixels[scene_indices_to_cluster_[neighbor_id]]++;
            else
                not_in_cluster++;

            inliers++;

        }
    }

    //cost per inlier
    h.eval.similarity_score /= (float)inliers;
    //measure clutter caused by incomplete explained clusters
    h.eval.clutter_score = 0;
    if(inliers - not_in_cluster <= 0)
        h.eval.clutter_score = 1;
    else{
        for(int cl=0; cl<scene_clusters_explained_pixels.size(); ++cl){
            if(scene_clusters_explained_pixels[cl] != 0){
                //not explained points of the cluster
                int non_explained_cluster_points =
                        scene_clusters_[cl].indices.size() - scene_clusters_explained_pixels[cl];
                h.eval.clutter_score +=
                        ((float)non_explained_cluster_points / (float)scene_clusters_[cl].indices.size()) *
                        ((float)model_clusters_explained_pixels[cl] / (float)(inliers - not_in_cluster));
            }
        }
    }

    //cost per inlier
    h.eval.inliers_ratio = (float)(inliers) / (float)visible_points.indices.size();
    h.eval.pose_score = pose_score;
    h.eval.location_score = location_hough_score;

    float final_score = h.eval.similarity_score * similarity_reg_ +
                        h.eval.inliers_ratio    * inliers_reg_    -
                        h.eval.clutter_score    * clutter_reg_    +
                        h.eval.pose_score       * pose_score_reg_ +
                        h.eval.location_score   * location_score_reg_;

    h.eval.final_score = final_score;


    if(h.eval.clutter_score > clutter_threshold_ || h.eval.inliers_ratio < inliers_threshold_)
        return false;
    else
        return (final_score > final_score_threshold_);
}

//quickly find next solution vector based on mutual exclusion map
//solution's least significant bit is at [0]
//solution is actually a binary number, so increase by 1
//if the last 1 inserted is mutual exclusive with another 1 in fron of it,
//do not consider solutions less than this number, make 1 -> 0 and continue to search from there
bool MeshUtils::get_next_solution_vector(std::vector<bool> &solution,
                              const std::vector<int> &hypotheses_group,
                              MutualExclusiveMap &mutual_exclusive_hypotheses)
{

    if(single_object_in_group_){

        int i;
        for(i=0; i<solution.size(); ++i)
            if(solution[i])
                break;

        if(i == solution.size()){
            solution[0] = true;
            return true;
        }
        if(i == solution.size()-1)
            return false;

        solution[i] = false;
        solution[i+1] = true;
        return true;


    } else {

        int pos = 0;
        while(true){
            //increase the *binary number* - solution
            bool found = false;
            for(int i=pos; i<solution.size(); ++i){
                if(solution[i] == true)
                    solution[i] = false;
                else{
                    solution[i] = true;
                    pos = i;
                    found = true;
                    break;
                }
            }
            //no other solution, all true (i.e. ones)
            if(!found)
                return false;

            bool valid_solution = true;
            for(int i=pos+1; i<solution.size(); ++i){
                if(solution[i]){
                    if(mutual_exclusive_hypotheses.count(hypotheses_group[pos])){
                        if(mutual_exclusive_hypotheses[hypotheses_group[pos]].count(hypotheses_group[i])){
                            valid_solution = false;
                            break;
                        }
                    }
                }
            }
            if(valid_solution)
                return true;
            solution[pos++] = false;
        }
    }

}


std::vector<int> MeshUtils::optimize_hypotheses_multi(std::vector<ObjectHypothesis> &hypotheses){

    std::vector<int> result;

    //check mutual exclusiveness between hypotheses. Will save time in optimizing.
    //some are already applied. Mutual exclusive are hypotheses
    //of the same location of the same object with different pose.
    MutualExclusiveMap mutual_exclusive_hypotheses;
    for(int i=0; i<hypotheses.size(); ++i){
        for(int j=i+1; j<hypotheses.size(); ++j){            

            //first check if centers are far away

            Eigen::Vector3f center_i(hypotheses[i].rotmat(0, 3), hypotheses[i].rotmat(1, 3), hypotheses[i].rotmat(2, 3));
            Eigen::Vector3f center_j(hypotheses[j].rotmat(0, 3), hypotheses[j].rotmat(1, 3), hypotheses[j].rotmat(2, 3));
            float dist = sqrt( pow(center_i(0) - center_j(0), 2.0f) +
                               pow(center_i(1) - center_j(1), 2.0f) +
                               pow(center_i(2) - center_j(2), 2.0f) );
            //if they are close enough
            int obj_id_i = hypotheses[i].obj_id;
            int obj_id_j = hypotheses[j].obj_id;
            if(dist < object_bounding_boxes_[obj_id_i].max_center_length + object_bounding_boxes_[obj_id_j].max_center_length){

                //check same explained points on scene
                int common_scene_points = 0;
                const boost::unordered_map<int, bool> *scene_points_i = &(hypotheses[i].eval.scene_explained_indices);
                const boost::unordered_map<int, bool> *scene_points_j = &(hypotheses[j].eval.scene_explained_indices);
                for(boost::unordered_map<int, bool>::const_iterator it=scene_points_i->begin(); it!=scene_points_i->end(); ++it)
                    if(scene_points_j->count(it->first))
                        common_scene_points++;
                if( (float)common_scene_points / (float)hypotheses[i].eval.total_scene_indices_explained > 0.4f ||
                    (float)common_scene_points / (float)hypotheses[j].eval.total_scene_indices_explained > 0.4f ){

                    mutual_exclusive_hypotheses[i][j] = true;
                    mutual_exclusive_hypotheses[j][i] = true;
                }

            }

        }
    }

    //display mutual exlusive hypotheses
//    for(MutualExclusiveMap::iterator tmp = mutual_exclusive_hypotheses.begin(); tmp != mutual_exclusive_hypotheses.end(); ++tmp){
//        boost::unordered_map<int, bool> *mt2 = &(tmp->second);
//        for(boost::unordered_map<int, bool>::iterator tmp2 = mt2->begin(); tmp2 != mt2->end(); ++tmp2){
//            cv::Mat rgb_tmp;
//            scene_rgb_.copyTo(rgb_tmp);
//            srand(123321);
//            cv::Scalar color = cv::Scalar(rand()%255, rand()%255, rand()%255);
//            draw_hypotheses_boundingbox(hypotheses[tmp->first], rgb_tmp, color);
//            cv::Scalar color2 = cv::Scalar(rand()%255, rand()%255, rand()%255);
//            draw_hypotheses_boundingbox(hypotheses[tmp2->first], rgb_tmp, color2);

//            int k=-1;
//            while(k==-1){
//                cv::imshow("rgb", rgb_tmp);
//                k = cv::waitKey(30);
//            }
//            if((char)k == 's')
//                break;

//        }
//    }

    //find groups of mutual exclusive hypotheses, i.e. connected graphs
    std::vector<bool> visited(hypotheses.size(), false);
    std::vector< std::vector<int> > hypotheses_groups;
    for(int i=0; i<hypotheses.size(); ++i){
        if(!visited[i]){
            visited[i] = true;
            std::queue<int> q;
            q.push(i);
            hypotheses_groups.push_back(std::vector<int>(1, i));
            while(!q.empty()){
                int cur_h = q.front();
                q.pop();
                for(int j=0; j<hypotheses.size(); ++j){
                    if(!visited[j]){
                        if(mutual_exclusive_hypotheses.count(cur_h)){
                            if(mutual_exclusive_hypotheses[cur_h].count(j)){

                                hypotheses_groups.back().push_back(j);
                                q.push(j);
                                visited[j] = true;
                            }
                        }
                    }
                }
            }
        }
    }



    std::vector<cv::Scalar> class_colors(4);
    class_colors[0] = cv::Scalar(255, 0, 0);
    class_colors[1] = cv::Scalar(0, 255, 0);
    class_colors[2] = cv::Scalar(0,0,255);
    class_colors[3] = cv::Scalar(255, 0, 255);

    //optimize each group separately
    cv::Mat solution_rgb;
    scene_rgb_.copyTo(solution_rgb);

    for(int g=0; g<hypotheses_groups.size(); ++g){

        //visualize group
//        cv::Mat rgb_tmp;
//        scene_rgb_.copyTo(rgb_tmp);
//        for(int i=0; i<hypotheses_groups[g].size(); ++i)
//            draw_hypotheses_boundingbox(hypotheses[hypotheses_groups[g][i]], rgb_tmp, class_colors[hypotheses[hypotheses_groups[g][i]].obj_id]);

//        int k=-1;
//        while(k==-1){
//            cv::imshow("rgb", rgb_tmp);
//            k = cv::waitKey(30);
//        }

        //normalize location score inside each group
        float group_best_location_score = 0;
        for(int h=0; h<hypotheses_groups[g].size(); ++h)
            if(group_best_location_score < hypotheses[hypotheses_groups[g][h]].eval.location_score)
                group_best_location_score = hypotheses[hypotheses_groups[g][h]].eval.location_score;


        //scene_occupied_pixels[scene_indice][hypothesis_index_inside_group]
        int total_scene_indices_explained_by_group = 0;
        boost::unordered_map<int, boost::unordered_map<int, bool> > scene_occupied_pixels;
        for(int i=0; i<hypotheses_groups[g].size(); ++i){
            const boost::unordered_map<int, bool> *scene_indices =
                    &(hypotheses[hypotheses_groups[g][i]].eval.scene_explained_indices);
            for(boost::unordered_map<int, bool>::const_iterator it=scene_indices->begin(); it!=scene_indices->end(); ++it){
                if(!scene_occupied_pixels.count(it->first))
                    total_scene_indices_explained_by_group++;
                scene_occupied_pixels[it->first][i] = true;
            }
        }


        std::vector<bool> solution(hypotheses_groups[g].size(), false);
        std::vector< std::vector<bool> > solutions;
        while(get_next_solution_vector(solution, hypotheses_groups[g], mutual_exclusive_hypotheses))
              solutions.push_back(solution);

        //std::cout << "group " << g << " solutions: " << solutions.size() << std::endl;

        std::vector<bool> best_solution(hypotheses_groups[g].size(), false);
        float best_score = 0;
        omp_set_num_threads(num_threads_);
        #pragma omp parallel
        {

            //store current solution
            std::vector<bool> best_solution_local(hypotheses_groups[g].size(), false);
            float best_score_local = 0;            

            #pragma omp for schedule(dynamic)
            for(int sol = 0; sol < solutions.size(); ++sol){

                std::vector<bool> *solution = &(solutions[sol]);

                int total_scene_indices_explained = 0;
                int common_indices_cost = 0;
                std::vector<int> cur_hypotheses;
                for(int s=0; s<solution->size(); ++s)
                    if((*solution)[s])
                        cur_hypotheses.push_back(s);


                for(boost::unordered_map<int, boost::unordered_map<int, bool> >::iterator it=scene_occupied_pixels.begin();
                    it != scene_occupied_pixels.end(); ++it){

                    int hypotheses_explained_current = 0;
                    for(int h=0; h<cur_hypotheses.size(); ++h)
                        if(it->second.count(cur_hypotheses[h]))
                            hypotheses_explained_current++;
                    if(hypotheses_explained_current > 0)
                        total_scene_indices_explained++;
                    if(hypotheses_explained_current > 1)
                        common_indices_cost += hypotheses_explained_current - 1;
                }

                HypothesisEvaluation avg_eval;
                for(std::vector<int>::iterator h=cur_hypotheses.begin(); h!=cur_hypotheses.end(); ++h){
                    avg_eval.clutter_score += hypotheses[hypotheses_groups[g][*h]].eval.clutter_score / (float)cur_hypotheses.size();
                    avg_eval.inliers_ratio += hypotheses[hypotheses_groups[g][*h]].eval.inliers_ratio / (float)cur_hypotheses.size();
                    avg_eval.similarity_score += hypotheses[hypotheses_groups[g][*h]].eval.similarity_score / (float)cur_hypotheses.size();
                    avg_eval.visibility_ratio += hypotheses[hypotheses_groups[g][*h]].eval.visibility_ratio / (float)cur_hypotheses.size();                    
                    avg_eval.location_score += hypotheses[hypotheses_groups[g][*h]].eval.location_score / (float)cur_hypotheses.size();
                    avg_eval.pose_score += hypotheses[hypotheses_groups[g][*h]].eval.pose_score / (float)cur_hypotheses.size();
                }

                float total_explained_ratio = (float)total_scene_indices_explained / (float)total_scene_indices_explained_by_group;
                float common_explained_ratio = (float)common_indices_cost/(float)total_scene_indices_explained_by_group;

                //get only one solution of the group if total_explained_ratio = 0
                float total_explained_ratio_reg, common_explained_ratio_reg;
                if(single_object_in_group_){
                    total_explained_ratio_reg = 0;
                    common_explained_ratio_reg = 0;
                } else {
                    total_explained_ratio_reg = total_explained_ratio_reg_;
                    common_explained_ratio_reg = common_explained_ratio_reg_;
                }


                float final_score = avg_eval.similarity_score * similarity_reg_ +
                                    avg_eval.inliers_ratio    * inliers_reg_    -
                                    avg_eval.clutter_score    * clutter_reg_    +
                                    avg_eval.pose_score       * pose_score_reg_ +
                                    avg_eval.location_score   * location_score_reg_ +
                                    total_explained_ratio     * total_explained_ratio_reg -
                                    common_explained_ratio    * common_explained_ratio_reg;


                if(final_score > best_score_local){
                    best_score_local = final_score;
                    best_solution_local = *solution;

                }

//                //visualize solution - need to comment out #pragma
//                //if(cur_hypotheses.size() > 0){
//        //            checkHypothesisWithGroundtruth(hypotheses[hypotheses_groups[g][cur_hypotheses[0]]]);
//                    std::cout << "total: " << total_scene_indices_explained << "  common: " << common_indices_cost <<std::endl;
//                    std::cout << "total ratio: " << total_explained_ratio << std::endl;
//                    std::cout << "common ratio: " << common_explained_ratio <<std::endl;

//                    std::cout << "clutter: " << avg_eval.clutter_score << std::endl;
//                    std::cout << "inliers: " << avg_eval.inliers_ratio << std::endl;
//                    std::cout << "location score: " << avg_eval.location_score << std::endl;
//                    std::cout << "pose score: " << avg_eval.pose_score << std::endl;
//                    std::cout << "similarity: " << avg_eval.similarity_score << std::endl;
//                    std::cout << "final score: " << final_score << std::endl;
//                    std::cout << "ground true: " << hypotheses[hypotheses_groups[g][cur_hypotheses[0]]].eval.ground_truth_error << std::endl;
//                    std::cout << std::endl;
//        //            std::cout << "h pose: " << std::endl << hypotheses[hypotheses_groups[g][cur_hypotheses[0]]].rotmat << std::endl;
//        //            std::cout << "g pose: " << std::endl << object_groundtruth_poses_[hypotheses[hypotheses_groups[g][cur_hypotheses[0]]].obj_id] << std::endl;
//                    cv::Mat rgb_tmp;
//                    scene_rgb_.copyTo(rgb_tmp);
//                    for(int i=0; i<solution->size(); ++i)
//                        if((*solution)[i])
//                            renderObject(rgb_tmp, hypotheses[hypotheses_groups[g][i]].obj_id, hypotheses[hypotheses_groups[g][i]].rotmat, 0.7);
//        //                    draw_hypotheses_boundingbox(hypotheses[hypotheses_groups[g][i]],
//        //                                                rgb_tmp,
//        //                                                class_colors[hypotheses[hypotheses_groups[g][i]].obj_id]);


//        //            cv::Mat gt_rgb;
//        //            scene_rgb_.copyTo(gt_rgb);
//        //            int obj_id_tmp = hypotheses[hypotheses_groups[g][0]].obj_id;
//        //            Eigen::Matrix4f gt_pose = object_groundtruth_poses_[obj_id_tmp][0];
//        //            for(int i=1; i<3; ++i)
//        //                for(int j=0; j<4; ++j)
//        //                    gt_pose(i, j) = -gt_pose(i, j);
//        //            renderObject(gt_rgb, obj_id_tmp, gt_pose, 0.7);

//                    int k=-1;
//                    while(k==-1){
//                        cv::imshow("rgb", rgb_tmp);
//        //                cv::imshow("ground truth", gt_rgb);
//                        k = cv::waitKey(30);
//                    }
//                    if((char)k == 's')
//                        break;
//                //}

            } //iterator of group solutions

            #pragma omp critical
            {
                if(best_score < best_score_local){
                    best_score = best_score_local;
                    best_solution = best_solution_local;
                }
            }

        }   //omp parallel

        for(int i=0; i<best_solution.size(); ++i){
            if(best_solution[i]){                
                result.push_back(hypotheses_groups[g][i]);
            }
        }

//        for(int i=0; i<best_solution.size(); ++i)
//            if(best_solution[i])
//                renderObject(solution_rgb, hypotheses[hypotheses_groups[g][i]].obj_id, hypotheses[hypotheses_groups[g][i]].rotmat, 0.7);



    }   //group iterator

//    int k=-1;
//    while(k==-1){
//        cv::imshow("solution", solution_rgb);
//        cv::imshow("ground truth", scene_rgb_);
//        k = cv::waitKey(30);
//    }


    return result;

}


std::vector<int> MeshUtils::optimize_hypotheses_single(std::vector<ObjectHypothesis> &hypotheses){


    if(hypotheses.size() == 0)
        return std::vector<int>(0);

    float best_score = 0;
    int best_solution = -1;

    int cur_solution = 0;
    for(std::vector<ObjectHypothesis>::iterator h = hypotheses.begin(); h != hypotheses.end(); ++h, ++cur_solution){

        float final_score = h->eval.similarity_score * similarity_reg_ +
                            h->eval.inliers_ratio    * inliers_reg_    -
                            h->eval.clutter_score    * clutter_reg_    +
                            h->eval.pose_score       * pose_score_reg_ ;

        h->eval.final_score = final_score;

        if(final_score > best_score){
            best_score = final_score;
            best_solution = cur_solution;
        }

        //visualize
//        checkHypothesisWithGroundtruth(*h);
//        std::cout << "clutter: " << h->eval.clutter_score << std::endl;
//        std::cout << "inliers: " << h->eval.inliers_ratio << std::endl;
//        std::cout << "pose score: " << h->eval.pose_score << std::endl;
//        std::cout << "location scoere: " << h->eval.location_score << std::endl;
//        std::cout << "similarity: " << h->eval.similarity_score << std::endl;
//        std::cout << "final score: " << final_score << std::endl;
//        std::cout << "ground true: " << h->eval.ground_truth_error;
//        std::cout << std::endl;
////        std::cout << "h pose: " << std::endl << hypotheses[hypotheses_groups[g][cur_hypotheses[0]]].rotmat << std::endl;
////        std::cout << "g pose: " << std::endl << object_groundtruth_poses_[hypotheses[hypotheses_groups[g][cur_hypotheses[0]]].obj_id] << std::endl;
//        cv::Mat rgb_tmp;
//        scene_rgb_.copyTo(rgb_tmp);
//        renderObject(rgb_tmp, h->obj_id, h->rotmat, 0.7);
//        cv::Mat gt_rgb;
//        scene_rgb_.copyTo(gt_rgb);
////        Eigen::Matrix4f gt_pose = object_groundtruth_poses_[h->obj_id][0];
////        for(int i=1; i<3; ++i)
////            for(int j=0; j<4; ++j)
////                gt_pose(i, j) = -gt_pose(i, j);
////        renderObject(gt_rgb, h->obj_id, gt_pose, 0.7);

//        int k=-1;
//        while(k==-1){
//            cv::imshow("rgb", rgb_tmp);
////            cv::imshow("ground truth", gt_rgb);
//            k = cv::waitKey(30);
//        }
//        if((char)k == 's')
//            break;

    }

    //checkHypothesisWithGroundtruth(hypotheses[best_solution]);
    std::vector<int> res(1, best_solution);

    return res;

}



//first evaluate_hypotheses should be called on each hypotheses
std::vector<int> MeshUtils::optimize_hypotheses(std::vector<ObjectHypothesis> &hypotheses){

    if(single_object_instance_)
        return optimize_hypotheses_single(hypotheses);
    else
        return optimize_hypotheses_multi(hypotheses);

}


