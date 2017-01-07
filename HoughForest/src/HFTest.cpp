#include <HFTest.h>
#include <cuda/surface_normals.h>
#include <cuda/patch_extractor.h>

#include <string>
#include <math.h>
#include <map>
#include <queue>
#include <utility>

#include <highgui.h>
#include <opencv2/contrib/contrib.hpp>


#include <glog/logging.h>

#include <caffe/proto/caffe.pb.h>


//3D point in xtion's camera frame, to 2d image coordinates
Eigen::Vector2i HFTest::Point3DToImage(const Eigen::Vector3f &p){

    Eigen::Vector2i res;

    if(p(2) == 0){
        res << 0,0;
        return res;
    }

    //get 2D image location of the center
    int res_x = p(0) / p(2) * fx_ + cx_ + 0.5f;
    int res_y = p(1) / p(2) * fy_ + cy_ + 0.5f;

    res << res_x, res_y;
    return res;

}

//return x, y in xtion coordinates as the center of the object
//from the dof vector of the leaf nodes
Eigen::Vector3f HFTest::get_obj_center_vote_from_6dof(Eigen::VectorXf dof,
                                                      int patch_x, int patch_y, unsigned short depth){

    //get camera prediction -> xtion_rotmat
    float cos_yaw = cos(dof(0));
    float sin_yaw = sin(dof(0));
    float cos_pitch = cos(dof(1));
    float sin_pitch = sin(dof(1));
    float cos_roll = cos(dof(2));
    float sin_roll = sin(dof(2));

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

    Eigen::Matrix4f rotmat = Rz * Ry * Rx;  


    //rotate rotmat (vtk camera) 180 degrees around X
    //to match xtion coordinate system
    Eigen::Matrix4f corrMat;
    corrMat << 1,  0,  0, 0,
               0, -1,  0, 0,
               0,  0, -1, 0,
               0,  0,  0, 1;
    Eigen::Matrix4f xtion_rotmat = corrMat * rotmat;    

    //put current patch as center of the world
    float z = (float)depth / 1000.0f;
    float x = ((float)patch_x - cx_)*z/fx_;
    float y = ((float)patch_y - cy_)*z/fy_;
    xtion_rotmat(0, 3) = x;
    xtion_rotmat(1, 3) = y;
    xtion_rotmat(2, 3) = z;
    //----------------------------------

    //get center vote in 3D space of xtion
    //center should be at -x -y -z
    //actually, if object is not centered, the vote is for the 0,0,0 of the
    //ply coordinate system
    Eigen::Vector4f center_obj;
    center_obj << -dof(3), -dof(4), -dof(5), 1;
    Eigen::Vector4f center_xtion = xtion_rotmat*center_obj;

    Eigen::Vector3f res;
    res << center_xtion(0), center_xtion(1), center_xtion(2);
    return res;
}



//align the z axis of an object hypotheses with the normal of the table
//coeffs contains the vertical vector pointing -above- the table
//(given by pcl segmentation
void HFTest::correctPose(Eigen::Matrix4f& objpose, Eigen::Vector4f coeffs){

    //make vector point above table
    if(coeffs(2) > 0) coeffs = -coeffs;

    Eigen::Vector3f n;
    n << coeffs(0), coeffs(1), coeffs(2);
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


bool hcomparator(const MeshUtils::ObjectHypothesis &h1, const MeshUtils::ObjectHypothesis &h2){
    return h1.eval.final_score > h2.eval.final_score;
}


HFTest::TreeNode* HFTest::get_leaf(TreeNode *node, const std::vector<float> &test_vec){

    if(node->leaf)
        return node;
    else{

        float val;
        if(node->test.measure_mode == 0)
            val = test_vec[node->test.feature1] - test_vec[node->test.feature2];
        else if(node->test.measure_mode == 1)
            val = test_vec[node->test.feature1];

        if(val < node->test.threshold)
            return get_leaf(node->left, test_vec);
        else
            return get_leaf(node->right, test_vec);

    }

}


void HFTest::detect(const std::vector<float> &test_vec,
                    int patch_x,
                    int patch_y,
                    unsigned short depth,
                    std::vector<float> &res,
                    std::vector<cv::Mat> &obj_center_hough_2d,                    
                    std::vector<VotesToNodeMap> &center_leaf_map,
                    const DetectorOptions::Options &detect_options)
{    

    res.resize(number_of_classes_, 0.0f);
    for(int t=0; t<trees_.size(); ++t){

        TreeNode *cur_leaf = get_leaf(trees_[t], test_vec);

        for(int c=0; c<number_of_classes_; ++c)
            res[c] += cur_leaf->class_prob[c] / (float)trees_.size();

        for(int c=0; c<detect_options.object_options_size(); c++) {
            if(!detect_options.object_options(c).should_detect()) continue;

            //---vote to houghmap of object center----
            //voting in x,y and then in z

            //allow hough votes from leafs with prob above a threshold
            if( cur_leaf->class_prob[c] >= 0.5f){
                for(int p=0; p<cur_leaf->hough_votes[c].size(); ++p){
                    //get 3d coords of object center from leaf vote
                    Eigen::Vector3f obj_center_3d =
                            get_obj_center_vote_from_6dof(cur_leaf->hough_votes[c][p], patch_x, patch_y, depth);
                    Eigen::Vector2i obj_center_2d = Point3DToImage(obj_center_3d);
                    //add vote with weight of the probability of the class
                    if( obj_center_2d(0) >= 0 &&
                        obj_center_2d(0) < obj_center_hough_2d[c].cols &&
                        obj_center_2d(1) >= 0 &&
                        obj_center_2d(1) < obj_center_hough_2d[c].rows){

                        obj_center_hough_2d[c].at<float>(obj_center_2d(1), obj_center_2d(0)) +=
                                cur_leaf->class_prob[c];
                    }

                    //add current leaf to center-leaf map for fast pose estimation later
                    HoughPoint h_2d(obj_center_2d(0), obj_center_2d(1), 0, 0, 0, 0);
                    if(!center_leaf_map[c].count(h_2d))
                        center_leaf_map[c][h_2d] = std::vector<TreeNode*>(0);
                    center_leaf_map[c][h_2d].push_back(cur_leaf);

                }
            }
        }
    }
}

void HFTest::non_max_suppression(const cv::Mat &input,
                                 std::vector<MapHypothesis> &local_maxima,
                                 cv::Size2i w){


    // double t = omp_get_wtime();

    int wx = w.width;
    int wy = w.height;

    std::vector<std::vector<std::pair<int, int> > > res(input.rows, std::vector<std::pair<int,int> >(input.cols - wx + 1));

    //horizontal scanning window
    for(int i=0; i<input.rows; ++i) {
        std::deque<std::pair<float, std::pair<int, int> > > q;
        for(int j=0; j<input.cols; ++j) {
            if(!q.empty() && q.front().second.second == j-wx)
                q.pop_front();
            float val = input.at<float>(i, j);
            while(!q.empty() && q.back().first < val)
                q.pop_back();
            q.push_back(std::make_pair(val, std::make_pair(i, j)));
            if(j >= wx-1) res[i][j-wx+1] = q.front().second;
        }
    }

    //vertical scanning window
    for(int j=0; j<input.cols - wx + 1; ++j) {
        std::deque<std::pair<float, std::pair<int, int> > > q;
        for(int i=0; i<input.rows - wy + 1; ++i) {
            if(!q.empty() && q.front().second.first == i-wy)
                q.pop_front();
            float val = input.at<float>(res[i][j].first, res[i][j].second);
            while(!q.empty() && q.back().first < val)
                q.pop_back();
            q.push_back(std::make_pair(val, std::make_pair(res[i][j].first, res[i][j].second)));
            if(i >= wy-1) {
                int cx = j + wx/2;
                int cy = (i-wy+1) + wy/2;
                if(q.front().first != 0 && q.front().second.first == cy && q.front().second.second == cx)
                    local_maxima.push_back(MapHypothesis(q.front().first, cv::Point(q.front().second.second, q.front().second.first)));
            }
        }
    }

    std::sort(local_maxima.begin(), local_maxima.end(), HypothesisComparator());

    //std::cout << "max suppress time: " << omp_get_wtime() - t << "ms" << std::endl;

}


void HFTest::get_leaf_map(TreeNode* n, LeafMap &leaf_map){

    if(n->leaf)
        leaf_map[n->leaf_id] = n;
    else{
        get_leaf_map(n->left, leaf_map);
        get_leaf_map(n->right, leaf_map);
    }

}



float HFTest::get_cam_dist(const Eigen::Vector3f &cam1, const Eigen::Vector3f &cam2){

    float dist = 0;
    for(int i=0; i<3; ++i)
        dist += acos( cos(cam1(i))*cos(cam2(i)) + sin(cam1(i))*sin(cam2(i)) );

    return dist;

}


DECLARE_bool(visualize_hypotheses);
std::vector<MeshUtils::ObjectHypothesis> HFTest::test_image(cv::Mat& rgb,  cv::Mat& depth,
             const DetectorOptions::Options &detect_options, MeshUtils &mesh_utils,
             bool generate_random_values, float patch_distance_threshold)
{


    ///////// GETTING HOUGH MAPS ///////////   


    int batch_size = batch_size_caffe_;


    CHECK(is_forest_loaded_) << "No forest is loaded";


    //TODO
    //We shouldn't load the network each time we test and image
    //get caffe net
    CHECK_GT(caffe_model_definition_filename_.size(), 0) << "No caffe definition model defined.";
    CHECK_GT(caffe_model_weights_filename_.size(), 0) << "No caffe weights model defined.";
    //caffe::Caffe::set_phase(caffe::Caffe::TEST);  // used with previous version of Caffe
    caffe::Net<float> caffe_net(caffe_model_definition_filename_, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(caffe_model_weights_filename_);   



/////// WITH NORMALS
///
//    //get surface normals
//    std::vector<unsigned short> depthvec;
//    depthvec.assign((unsigned short*)depth.datastart, (unsigned short*)depth.dataend);
//    std::vector<float> normals;
//    //Heigh x width x 3 (3 is the fastest moving, it is x y z)
//    surface_normals_gpu::generate_normals(depthvec, depth.cols, depth.rows, 575.0f, normals);


//    //create 3D texture vector Rows x Cols x 7 (B, G, R, D, (X, Y, Z)normals )
//    std::vector<float> texture_3D(rgb.rows * rgb.cols * 7);
//    int tex_pos = 0;
//    int norm_pos = 0;
//    for(int row=0; row<rgb.rows; ++row ){
//        for(int col=0; col<rgb.cols; ++col){
//            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[0] / 255.0f;  //B
//            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[1] / 255.0f;  //G
//            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[2] / 255.0f;  //R
//            texture_3D[tex_pos++] = (float)depth.at<unsigned short>(row, col);       //D (in mm)
//            texture_3D[tex_pos++] = normals[norm_pos++];                             //X norm
//            texture_3D[tex_pos++] = normals[norm_pos++];                             //Y norm
//            texture_3D[tex_pos++] = normals[norm_pos++];                             //Z norm
//        }
//    }


//    //extract patches
//    std::vector<float> patches;
//    //patch locations - [x1, y1, x2, y2 .. ]
//    std::vector<int> patches_loc;
//    patch_extractor_gpu::extract_patches(texture_3D,
//                                         rgb.cols,
//                                         rgb.rows,
//                                         patch_size_in_voxels_,
//                                         voxel_size_in_m_,
//                                         stride_in_pixels_,
//                                         575.0f,
//                                         patches,
//                                         patches_loc,
//                                         generate_random_values,
//                                         patch_distance_threshold);





    //create 3D texture vector Rows x Cols x 4 (B, G, R, D )
    std::vector<float> texture_3D(rgb.rows * rgb.cols * 4);
    int tex_pos = 0;
    for(int row=0; row<rgb.rows; ++row ){
        for(int col=0; col<rgb.cols; ++col){
            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[0] / 255.0f;  //B
            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[1] / 255.0f;  //G
            texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[2] / 255.0f;  //R
            texture_3D[tex_pos++] = (float)depth.at<unsigned short>(row, col);       //D (in mm)
        }
    }


    //extract patches
    std::vector<float> patches;
    //patch locations - [x1, y1, x2, y2 .. ]
    std::vector<int> patches_loc;
    patch_extractor_gpu::extract_patches_rgbd(texture_3D,
                                         rgb.cols,
                                         rgb.rows,
                                         patch_size_in_voxels_,
                                         voxel_size_in_m_,
                                         max_depth_range_in_m_,
                                         stride_in_pixels_,
                                         1.0f,
                                         fx_,
                                         patches,
                                         patches_loc,
                                         generate_random_values,
                                         patch_distance_threshold);



    //write result image
    cv::Mat res_img(rgb.rows, rgb.cols, CV_8UC3, cv::Scalar(0,0,0));
    std::vector<cv::Vec3b> class_colors(number_of_classes_);
    srand(123);
    for(int c=0; c<number_of_classes_; ++c)
        class_colors[c] = cv::Vec3b(rand()%256, rand()%256, rand()%256);


    Eigen::MatrixXi patch_classes(rgb.rows, rgb.cols);
    for(int i=0; i<rgb.rows; ++i)
        for(int j=0; j<rgb.cols; ++j)
            patch_classes(i, j) = -1;    

    //houghmap for every class
    std::vector<cv::Mat> class_hough_centers(number_of_classes_);
    for(int c=0; c<number_of_classes_; ++c)
        class_hough_centers[c] = cv::Mat (rgb.rows, rgb.cols, CV_32FC1, cv::Scalar(0));

    //from hough centers back to leafs for pose estimation
    std::vector<VotesToNodeMap> center_leaf_map(number_of_classes_);


    //when normals divide by 6
    //int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 6;
    int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 4;
    std::cout << "Number of patches: " << num_patches << std::endl;

    //batch input in caffe network    
    //TODO
    //We may loose some patches in the end if the last
    //batch is incomplete
    int num_batches = num_patches / batch_size;

    //for visualizaion
    cv::Mat patch_rgb(patch_size_in_voxels_, patch_size_in_voxels_, CV_8UC3);

    std::cout << "Extracting Patches ..." << std::endl;
    for(int b=0; b<num_batches; ++b){

        //std::cout << "Calculating patches: " << (b+1)*batch_size << " / " << num_patches << std::endl;

////WITH NORMALRS
//        std::vector<float> float_data(patch_size_in_voxels_ * patch_size_in_voxels_ * 6 * batch_size);
//        int pos=0;
//        for(int i=0; i<batch_size; ++i){


//            for(int c=0; c<6; ++c){
//                for(int row=0; row<patch_size_in_voxels_; ++row){
//                    for(int col=0; col<patch_size_in_voxels_; ++col){
//                        int idx = b * batch_size * patch_size_in_voxels_ * patch_size_in_voxels_ * 6 +
//                                  i * patch_size_in_voxels_ * patch_size_in_voxels_ * 6 +
//                                  row * patch_size_in_voxels_ * 6 +
//                                  col * 6 +
//                                  c;

//                        //surface normals are also quantized into 256 bins.
//                        //simulate the quantization so multiple by 255 -make char- then divide by 255.0f
//                        if(c<3){
//                            float_data[pos++] = (float)(static_cast<unsigned char>(patches[idx] * 255.0f)) / 255.0f;
//                            //patch_rgb.at<cv::Vec3b>(row, col)(c) = static_cast<unsigned char>(patches[idx] * 255.0f);
//                        }
//                        else
//                            float_data[pos++] = (float)(static_cast<unsigned char>( (patches[idx]/2.0 + 0.5f) * 255.0f)) / 255.0f;

//                    }
//                }
//            }
//        }



        //WITHOUT NORMALIZATION

//        std::vector<float> float_data(patch_size_in_voxels_ * patch_size_in_voxels_ * 4 * batch_size);
//        int pos=0;
//        for(int i=0; i<batch_size; ++i){


//            for(int c=0; c<4; ++c){
//                for(int row=0; row<patch_size_in_voxels_; ++row){
//                    for(int col=0; col<patch_size_in_voxels_; ++col){
//                        int idx = b * batch_size * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
//                                  i * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
//                                  row * patch_size_in_voxels_ * 4 +
//                                  col * 4 +
//                                  c;


//                         float_data[pos++] = (float)(static_cast<unsigned char>(patches[idx] * 255.0f)) / 255.0f;
//                    }
//                }
//            }
//        }


 //LOCAL NORMALIZATION

        std::vector<float> float_data;        
        for(int i=0; i<batch_size; ++i){

            //get mean
            std::vector<float> buf(patch_size_in_voxels_ * patch_size_in_voxels_ * 4);
            int pos=0;
            float mean_rgb = 0;
            float mean_depth = 0;
            for(int c=0; c<4; ++c){
                for(int row=0; row<patch_size_in_voxels_; ++row){
                    for(int col=0; col<patch_size_in_voxels_; ++col){
                        int idx = b * batch_size * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
                                  i * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
                                  row * patch_size_in_voxels_ * 4 +
                                  col * 4 +
                                  c;

                        buf[pos] = patches[idx];
                        if(c < 3)
                            mean_rgb += buf[pos] / (patch_size_in_voxels_ * patch_size_in_voxels_ * 3);
                        else
                            mean_depth += buf[pos] / (patch_size_in_voxels_ * patch_size_in_voxels_);
                        pos++;

                    }
                }
            }

            //get std
            float std_rgb = 0;
            float std_depth = 0;
            for(int j=0; j<buf.size(); ++j){
                if(j < patch_size_in_voxels_ * patch_size_in_voxels_ * 3)
                    std_rgb += pow(buf[j] - mean_rgb, 2)/(patch_size_in_voxels_ * patch_size_in_voxels_ * 3);
                 else
                    std_depth += pow(buf[j] - mean_depth, 2)/(patch_size_in_voxels_ * patch_size_in_voxels_);
            }



            for(int j=0; j<buf.size(); ++j){
                //Truncate to +/-3 standard deviations and scale to -1 to 1
                if(j < patch_size_in_voxels_ * patch_size_in_voxels_ * 3){
                    buf[j] -= mean_rgb;
                    if(buf[j] > 3*std_rgb)
                        buf[j] = 3*std_rgb;
                    if(buf[j] < -3*std_rgb)
                        buf[j] = -3*std_rgb;
                    buf[j] /= 3*std_rgb;
                }
                else{
                    buf[j] -= mean_depth;
                    if(buf[j] > 3*std_depth)
                        buf[j] = 3*std_depth;
                    if(buf[j] < -3*std_depth)
                        buf[j] = -3*std_depth;
                    buf[j] /= 3*std_depth;

                }


                // Rescale from [-1,1] to [0.1,0.9]
                buf[j] = (buf[j] + 1) * 0.4f + 0.1f;

                //imitate quantization
                buf[j] = (float)(static_cast<unsigned char>(buf[j] * 255.0f)) / 255.0f;
            }

            float_data.insert(float_data.end(), buf.begin(), buf.end());

        }



        // for patch visualization

//        int k=-1;
//        while(k == -1){
//            cv::imshow("patch_rgb", patch_rgb);
//            k = cv::waitKey(30);

//        }



        if(use_gpu_)
            caffe::caffe_copy(caffe_net.input_blobs()[0]->count(), &(float_data[0]), caffe_net.input_blobs()[0]->mutable_gpu_data());
        else
            caffe::caffe_copy(caffe_net.input_blobs()[0]->count(), &(float_data[0]), caffe_net.input_blobs()[0]->mutable_cpu_data());

        //Get feature from Net - forward image patch
        //output_blob[0]: contains the result
        //output_blob[1]: contains the input
        const std::vector<caffe::Blob<float>* > output_blob = caffe_net.ForwardPrefilled();

        int feature_vector_length = output_blob[0]->count() / batch_size;
        CHECK_EQ(feature_vector_length, feature_vector_length_) << "Output vector size of caffe net and Input vector size of forest do not match";



        omp_set_num_threads(num_threads_);
        #pragma omp parallel
        {           

            //houghmap for every class
            std::vector<cv::Mat> class_hough_centers_local(number_of_classes_);
            for(int c=0; c<number_of_classes_; ++c)
                class_hough_centers_local[c] = cv::Mat (rgb.rows, rgb.cols, CV_32FC1, cv::Scalar(0));

            //from hough centers back to leafs for pose estimation
            std::vector<VotesToNodeMap> center_leaf_map_local(number_of_classes_);

            #pragma omp for schedule(dynamic)
            for(int i=0; i<batch_size; ++i){                                                

                //get output of net for each patch of the batch
                std::vector<float> test_vec(feature_vector_length);


                for(int f=0; f<feature_vector_length; ++f)
                    test_vec[f] = output_blob[0]->data_at(i, f, 0, 0);                    

                int patch_idx = b * batch_size + i;
                int patch_x = patches_loc[2*patch_idx];
                int patch_y = patches_loc[2*patch_idx +1];

                //pass feature vectors down the forest
                std::vector<float> res(number_of_classes_);
                detect(test_vec, patch_x, patch_y, depth.at<unsigned short>(patch_y, patch_x),
                       res, class_hough_centers_local, center_leaf_map_local, detect_options);

                int best_class = -1;
                float best_prob = 0;
                for(int c=0; c<number_of_classes_; ++c){
                    if(res[c] > best_prob){
                        best_prob = res[c];
                        best_class = c;
                    }
                }
                patch_classes(patch_y, patch_x) = best_class;

                res_img.at<cv::Vec3b>(patch_y, patch_x) = class_colors[best_class];

            }

            #pragma omp critical
            {
                for(int c=0; c<number_of_classes_; ++c){
                    class_hough_centers[c] += class_hough_centers_local[c];
                    for(VotesToNodeMap::iterator it=center_leaf_map_local[c].begin(); it!=center_leaf_map_local[c].end(); ++it){
                        center_leaf_map[c][it->first].insert(center_leaf_map[c][it->first].end(), it->second.begin(), it->second.end());
                    }
                }                

            }

        }

    }
    std::cout << std::endl;


//    check_results(center_leaf_map, mesh_utils, patch_classes);

//    int kk=-1;
//    while(kk==-1){
//        cv::imshow("classification", res_img);
//        cv::imshow("rgb", rgb);
//        kk = cv::waitKey(30);
//    }


    ////////// CALCULATING HYPOTHESES /////////////


    int max_yaw_pitch_hypotheses = max_yaw_pitch_hypotheses_;
    int max_roll_hypotheses = max_roll_hypotheses_;
    float min_location_score_ratio = min_location_score_ratio_;
    float min_yaw_pitch_drop_ratio = min_yaw_pitch_drop_ratio_;


    //----Get hough maps of the object centers----
    //for each center make a hough voting for the pose   

    int centers_blur_size = centers_blur_size_;
    int centers_maxsupression_wsize = centers_maxsupression_wsize_;

    int pose_2d_blur_size = pose_2d_blur_size_;
    int pose_2d_maxsupression_wsize = pose_2d_maxsupression_wsize_;


    //store all valid hypotheses
    std::vector<MeshUtils::ObjectHypothesis> object_hypotheses;

    for(int c=0; c<detect_options.object_options_size(); c++) {
        if(!detect_options.object_options(c).should_detect()) continue;

        std::cout << "Generating Hypotheses for class: " <<
                     detect_options.object_options(c).name() << std::endl;
        double obj_class_exec_time = omp_get_wtime();        

        //blur centers votes to mix up
        cv::blur(class_hough_centers[c], class_hough_centers[c], cv::Size(centers_blur_size, centers_blur_size));

        //get local maxima
        std::vector<MapHypothesis> centers_hypotheses;
        non_max_suppression(class_hough_centers[c], centers_hypotheses,
                            cv::Size2i(centers_maxsupression_wsize, centers_maxsupression_wsize));

        //for each center hypothesis find pose        
        int max_loc_h = std::min(detect_options.object_options(c).max_location_hypotheses(),
                                 (int)centers_hypotheses.size());
        std::cout << "max locations: " << max_loc_h << std::endl;

        if (FLAGS_visualize_hypotheses && num_threads_ > 1) {
            std::cout << "Setting number of threads 1 due to visualization" << std::endl;
            omp_set_num_threads(1);
        }
        #pragma omp parallel
        {
            std::vector<MeshUtils::ObjectHypothesis> object_hypotheses_local;

            #pragma omp for schedule(dynamic)
            for(int center_hypotheses = 0; center_hypotheses < max_loc_h; ++center_hypotheses){

                float location_hough_score = centers_hypotheses[center_hypotheses].first;
                if( location_hough_score / centers_hypotheses[0].first < min_location_score_ratio)
                    continue;
                //std::cout << "location score: " << location_hough_score << std::endl;
                cv::Point cur_center = centers_hypotheses[center_hypotheses].second;

                //visualize maxima
//                cv::circle(rgb, cv::Point(cur_center.x, cur_center.y), 7, cv::Scalar(class_colors[c](0),class_colors[c](1),class_colors[c](2)), 3);
//                cv::circle(class_hough_centers[c], cv::Point(cur_center.x, cur_center.y), 10, cv::Scalar(1), 5);

//                while(cv::waitKey(30)==-1)
//                    cv::imshow("temp", rgb);


                //search inside a window of centers_maxsupression_wsize
                //to find mean z                

                float mode_z = 0;
                float z_bin_size = 0.01f; // bin size of z quantization
                float max_z = 3.0f; // max z limit for centers
                int z_maxsuppression_wsize = 20;
                cv::Mat houghmap_z( (int)(max_z/z_bin_size), 1, CV_32FC1, cv::Scalar(0));                

                //get 3D hough votes for pose
                //currently commended because 2d pose voting worked a bit better
                //Hough3DMap pose_houghmap;

                //vote first for yaw-pitch and keep roll nodemap to calc roll later
                //votes are from -pi to pi, but we vote also in -2pi,2pi for boundary effects in voting
                VotesToNodeMap roll_nodemap;
                cv::Mat pose_houghmap_2d(720, 720, CV_32FC1, cv::Scalar(0)); //[-360,359]

                for(int row=cur_center.y-centers_maxsupression_wsize/2;
                    row<cur_center.y+centers_maxsupression_wsize/2; ++row){

                    for(int col=cur_center.x-centers_maxsupression_wsize/2;
                        col<cur_center.x+centers_maxsupression_wsize/2; ++col){

                        //get leafs that voted for this pixel center
                        HoughPoint hp(col, row, 0, 0, 0, 0);
                        if(center_leaf_map[c].count(hp)){
                            std::vector<TreeNode*> *cur_votes = &(center_leaf_map[c][hp]);
                            //get mode of z
                            for(int v=0; v<cur_votes->size(); ++v){
                                for(int p=0; p<(*cur_votes)[v]->hough_votes[c].size(); ++p){
                                    if(depth.at<unsigned short>(row,col) != 0){
                                        Eigen::Vector3f obj_center = get_obj_center_vote_from_6dof((*cur_votes)[v]->hough_votes[c][p],
                                                                                                   col, row, depth.at<unsigned short>(row,col));                                        
                                        int z_bin = obj_center(2) / z_bin_size;
                                        if(z_bin < houghmap_z.rows && z_bin >= 0)
                                            houghmap_z.at<float>(z_bin) += (*cur_votes)[v]->class_prob[c];
                                    }                                    

                                    //2d pose voting in yaw-pitch, quantize to degrees
                                    int yaw   = (*cur_votes)[v]->hough_votes[c][p](0) / M_PI * 180.0f;
                                    int pitch = (*cur_votes)[v]->hough_votes[c][p](1) / M_PI * 180.0f;
                                    //vote also for +-360 accordingly for boundary effects in angles
                                    for(int k1=0; k1<2; ++k1){
                                        for(int k2=0; k2<2; ++k2){
                                            float sign_yaw = copysign(1, (float)yaw);
                                            float sign_pitch = copysign(1, (float)pitch);
                                            int cur_yaw = yaw + (-1)*(int)sign_yaw*k1*360 + 360;
                                            int cur_pitch = pitch + (-1)*(int)sign_pitch*k2*360 + 360;
                                            pose_houghmap_2d.at<float>(cur_yaw, cur_pitch) += (*cur_votes)[v]->class_prob[c];
                                            HoughPoint roll_hp(cur_yaw, cur_pitch, 0, 0, 0, 0);
                                            if(k1==0 && k2==0){
                                                //keep leaf for voting roll later
                                                if(!roll_nodemap.count(roll_hp))
                                                    roll_nodemap[roll_hp] = std::vector<TreeNode*>(0);
                                                roll_nodemap[roll_hp].push_back((*cur_votes)[v]);
                                            }
                                        }
                                    }                                    
                                }
                            }
                        }
                    }
                }                
                std::vector<MapHypothesis> z_hypotheses;
                non_max_suppression(houghmap_z, z_hypotheses, cv::Size2i(1, z_maxsuppression_wsize));
                if(z_hypotheses.size() != 0){
                    cv::Point best_z = z_hypotheses[0].second;
                    mode_z = best_z.y * z_bin_size;
                }                
                else{
                    //No hypothesis for z found ...
                    continue;
                }               


                //-----pose hypotheses 2d

                cv::blur(pose_houghmap_2d, pose_houghmap_2d, cv::Size(pose_2d_blur_size, pose_2d_blur_size));
                std::vector<MapHypothesis> pose_hypotheses_2d;

                //get local maxima
                non_max_suppression(pose_houghmap_2d, pose_hypotheses_2d,
                                    cv::Size2i(pose_2d_maxsupression_wsize, pose_2d_maxsupression_wsize));

                std::vector<MapHypothesis>::iterator it_p2d = pose_hypotheses_2d.begin();
                //keep solutions between -180,180
                while(it_p2d != pose_hypotheses_2d.end()){
                    if( (it_p2d->second).x < 180 ||
                        (it_p2d->second).x > 360+180 ||
                        (it_p2d->second).y < 180 ||
                        (it_p2d->second).y > 360 + 180 )

                        it_p2d = pose_hypotheses_2d.erase(it_p2d);
                    else
                        it_p2d++;

                }

                //iterate though yaw-pitch maxima and find roll maxima
                //if hypotheses score drops more than best_yaw_pitch_score_ratio of best stop..

                int max_yawpitch_h = std::min(max_yaw_pitch_hypotheses, (int)pose_hypotheses_2d.size());
                for(int h_2d=0; h_2d<max_yawpitch_h; ++h_2d){

                    float cur_yawpitch_score = pose_hypotheses_2d[h_2d].first / pose_hypotheses_2d[0].first;
                    if(cur_yawpitch_score < min_yaw_pitch_drop_ratio){
                        //std::cout << "yaw-pitch hypotheses dropped more than " << best_yaw_pitch_score_ratio << std::endl;
                        break;
                    }

                    //get hough votes for roll for current yaw-pitch
                    cv::Point cur_yawpitch = pose_hypotheses_2d[h_2d].second;
                    cv::Mat pose_houghmap_roll(720, 1, CV_32FC1, cv::Scalar(0)); //[-360,359]
                    for(int row=cur_yawpitch.y-pose_2d_blur_size/2; row<cur_yawpitch.y+pose_2d_blur_size/2; ++row){
                        for(int col=cur_yawpitch.x-pose_2d_blur_size/2; col<cur_yawpitch.x+pose_2d_blur_size/2; ++col){
                            int cur_yaw = row;
                            int cur_pitch = col;
                            HoughPoint roll_hp(cur_yaw, cur_pitch, 0, 0, 0, 0);
                            if(roll_nodemap.count(roll_hp)){
                                std::vector<TreeNode*> *cur_node_list = &(roll_nodemap[roll_hp]);
                                for(int node=0; node<cur_node_list->size(); ++node){
                                    TreeNode *cur_node = (*cur_node_list)[node];
                                    for(int p=0; p<cur_node->hough_votes[c].size(); ++p){
                                        int cur_roll = (cur_node->hough_votes[c][p])(2) * 180.0f / M_PI;
                                        pose_houghmap_roll.at<float>(cur_roll + 360) += cur_node->class_prob[c];
                                        if(cur_roll < 0)
                                            pose_houghmap_roll.at<float>(cur_roll + 360 + 360) += cur_node->class_prob[c];
                                        else
                                            pose_houghmap_roll.at<float>(cur_roll - 360 + 360) += cur_node->class_prob[c];
                                    }
                                }
                            }
                        }
                    }
                    cv::blur(pose_houghmap_roll, pose_houghmap_roll, cv::Size(1, pose_2d_blur_size));

                    //get local maxima of roll
                    std::vector<MapHypothesis> roll_hypotheses;
                    non_max_suppression(pose_houghmap_roll, roll_hypotheses, cv::Size2i(1, pose_2d_maxsupression_wsize));

                    //visualize
//                    double min_val,max_val;
//                    cv::minMaxLoc(pose_houghmap_roll, &min_val, &max_val);
//                    pose_houghmap_roll /= max_val;
//                    cv::Mat tmp_roll(720, 100, CV_32FC1);
//                    for(int k1=0; k1<720; k1++)
//                        for(int k2=0; k2<100; k2++)
//                            tmp_roll.at<float>(k1, k2) = pose_houghmap_roll.at<float>(k1, 0);
//                    int k=-1;
//                    while(k==-1){
//                        cv::imshow("roll", tmp_roll);
//                        k = cv::waitKey(30);
//                    }

                    //keep solutions in range -180,180
                    std::vector<MapHypothesis>::iterator it_proll = roll_hypotheses.begin();
                    while(it_proll != roll_hypotheses.end()){
                        if( (it_proll->second).y < 180 || (it_proll->second).y > 360 + 180 )
                            it_proll = roll_hypotheses.erase(it_proll);
                        else
                            it_proll++;
                    }


                    std::vector<MapHypothesis>::iterator it_roll = roll_hypotheses.begin();
                    int h_roll = 0;
                    float prev_h_roll = FLT_MAX;
                    while(h_roll < max_roll_hypotheses && it_roll != roll_hypotheses.end()){

    //                    std::cout << std::endl;
    //                    std::cout << "class: " << c << std::endl;
    //                    std::cout << "location hypothesis: " << center_hypotheses << "  angle hypothesis: " << h_2d << std::endl;
    //                    std::cout << "---------------" << std::endl;

                        float cur_roll_score = it_roll->first / roll_hypotheses[0].first;
                        cv::Point roll_p = it_roll->second;
                        //because some roll hypotheses are close to each other
                        //we take only those that are 7 degrees apart from the previous one
                        float dot = cos(prev_h_roll/180.0f * M_PI) * cos(roll_p.y/180.0f * M_PI) +
                                    sin(prev_h_roll/180.0f * M_PI) * sin(roll_p.y/180.0f * M_PI);

                        if(h_roll == 0 || acos(dot)/M_PI*180.0f > 7){
                            float yaw = (float)(cur_yawpitch.y - 360) / 180.0f * M_PI; //yaw
                            float pitch = (float)(cur_yawpitch.x - 360) / 180.0f * M_PI; //pitch
                            float roll = (float)(roll_p.y - 360) / 180.0f * M_PI; //roll                            

                            Eigen::Matrix4f rotmat;                            
                            mesh_utils.icp(c, cur_center.y, cur_center.x, mode_z, yaw, pitch, roll,
                                           rotmat, detect_options.object_options(c).icp_iterations());


                            for(int add_z = 0; add_z < 1; ++add_z){

                                MeshUtils::ObjectHypothesis h(c, rotmat);
                                bool accepted = mesh_utils.evaluate_hypothesis(h, location_hough_score, (cur_yawpitch_score + cur_roll_score) / 2.0f);

                                if (FLAGS_visualize_hypotheses) {

                                    // avoid hypotheses with high clutter score
                                    if(h.eval.clutter_score < 0.7) {
                                        std::cout << "accepted: " << accepted << std::endl;
                                        std::cout << "inliers: " << h.eval.inliers_ratio << std::endl;
                                        std::cout << "clutter: " << h.eval.clutter_score << std::endl;
                                        std::cout << "similarity: " << h.eval.similarity_score << std::endl;
                                        std::cout << "pose score: " << h.eval.pose_score << std::endl;
                                        std::cout << "location score: " << h.eval.location_score << std::endl;
                                        std::cout << "final score: " << h.eval.final_score << std::endl;
                                        std::cout << std::endl;

                                        cv::Mat rgb_rendered;
                                        rgb.copyTo(rgb_rendered);
                                        mesh_utils.renderObject(rgb_rendered, c, rotmat, 0.7f);

                                        while(cv::waitKey(30) == -1)
                                            cv::imshow("hypothesis visualization", rgb_rendered);
                                    } else {
					std::cout << "Not showing hypothesis due to high clutter score" << std::endl;
				    }
                                }

                                //threshold hypothesis                                
                                if(accepted)
                                    object_hypotheses_local.push_back(h);
                            }                            

                            prev_h_roll = roll_p.y;
                            h_roll++;
                        }

                        it_roll++;

                    } //iterate roll hypotheses

                } //iterate yaw-pitch hypotheses

            } // center locations hypotheses parallel for iteration

            #pragma omp critical
            {                
                object_hypotheses.insert(object_hypotheses.end(), object_hypotheses_local.begin(), object_hypotheses_local.end());                
            }

        } //parallel section of center hyotheses

        //std::cout << "Total hypotheses found: " << object_hypotheses.size() << std::endl;
        //std::cout << "Total time on icp: " << icp_total_time << "sec" << std::endl;
        std::cout << "Total execution time: " << omp_get_wtime() - obj_class_exec_time << "sec" << std::endl;
    }


    std::sort(object_hypotheses.begin(), object_hypotheses.end(), hcomparator);
    std::vector<int> solution = mesh_utils.optimize_hypotheses(object_hypotheses);
    std::vector<MeshUtils::ObjectHypothesis> best_h;
    for(int i=0; i<solution.size(); ++i)
        best_h.push_back(object_hypotheses[solution[i]]);



    //visualize hough maps
//    for(int c=0; c<detect_options.object_options_size(); c++) {
//        if(!detect_options.object_options(c).should_detect()) continue;

//        //normalize for visualization purposes
//        double min_val, max_val;
//        cv::Point min_loc, max_loc;
//        cv::minMaxLoc(class_hough_centers[c], &min_val, &max_val, &min_loc, &max_loc);
//        class_hough_centers[c] /= max_val;
//    }

//    while(cv::waitKey(30) == -1){
//        cv::imshow("rgb", rgb);
//        cv::imshow("res", res_img);

//        cv::imshow("hough centers of class 0", class_hough_centers[0]);

//    }



    return best_h;
}


//yaw pitch roll --> rads!
void get_yaw_pitch_roll_from_rot_mat(const Eigen::Matrix4f &rot_mat, float &yaw, float &pitch, float &roll){

    yaw = atan2(rot_mat(1, 0), rot_mat(0, 0));
    float a = sqrt(rot_mat(2, 1)*rot_mat(2, 1) + rot_mat(2, 2)*rot_mat(2, 2));
    pitch = atan2(-rot_mat(2,0), a);
    roll = atan2(rot_mat(2,1), rot_mat(2,2));

}

// was used for calculating results, but currently in this version
// it is not used. It's left here in case it is needed.
void HFTest::check_results(std::vector<VotesToNodeMap> &center_leaf_map, MeshUtils &mesh_utils, const Eigen::MatrixXi &patch_classes){


    std::vector<float> class_results;
    std::vector<float> reg_results;

    for(int c=0; c<number_of_classes_; ++c){

        int total_points = 0;
        float total_poses = 0;
        int correctly_classified = 0;
        float pose_error = 0;

        //currently support only one object in the scene
        Eigen::Matrix4f ground_truth_rotmat = mesh_utils.getGroundTruthPose(c, 0);
        Eigen::Vector3f ground_truth_pose;
        get_yaw_pitch_roll_from_rot_mat(ground_truth_rotmat, ground_truth_pose(0), ground_truth_pose(1), ground_truth_pose(2));


        cv::Mat mask = mesh_utils.getObjMask(c, mesh_utils.getGroundTruthPose(c, 0));
        int size = 5;
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                               cv::Size( 2*size + 1, 2*size+1 ),
                                               cv::Point( size, size ) );
        cv::erode(mask, mask, element);
        cv::dilate(mask, mask, element);


        for(int row=0; row<mask.rows; ++row){
            for(int col=0; col<mask.cols; ++col){
                if(mask.at<uchar>(row, col) == 1){

                    if(patch_classes(row, col) != -1) {
                        total_points ++;
                        if( patch_classes(row, col) == c ||
                            patch_classes(row+1, col) == c ||
                            patch_classes(row+1, col+1) == c ||
                            patch_classes(row, col+1) == c ||
                            patch_classes(row-1, col+1) == c ||
                            patch_classes(row-1, col) == c ||
                            patch_classes(row-1, col-1) == c ||
                            patch_classes(row, col-1) == c ||
                            patch_classes(row+1, col-1) == c)

                            correctly_classified ++;
                        else
                            mask.at<uchar>(row, col) = 0;

                        HoughPoint hp(col, row, 0, 0, 0, 0);

                        std::vector<TreeNode*> *leafs = &(center_leaf_map[c][hp]);

                        for(int l=0; l<leafs->size(); ++l){

                            for(int h=0; h<(*leafs)[l]->hough_votes[c].size(); ++h){
                                Eigen::VectorXf *pose = &((*leafs)[l]->hough_votes[c][h]);
                                float weight = (*leafs)[l]->class_prob[c];
                                pose_error += weight * sqrt(pow((*pose)(0) - ground_truth_pose(0), 2) +
                                                   pow((*pose)(1) - ground_truth_pose(1), 2) +
                                                   pow((*pose)(2) - ground_truth_pose(2), 2) );

                                total_poses += weight;

                            }

                        }
                    }
                }
            }
        }

        float class_result = (float)correctly_classified / (float)total_points;
        float reg_result = pose_error / (float)total_poses ;

        class_results.push_back(class_result);
        reg_results.push_back(reg_result);

//        std::cout << "Object: " << c << std::endl;
//        std::cout << "classification: " << class_result << std::endl;
//        std::cout << "regression: " << reg_result << std::endl;
//        std::cout << std::endl;

//        int k=-1;
//        while(k==-1){
//            cv::imshow("mask", mask*255);
//            k = cv::waitKey(30);
//        }
    }

    std::ofstream fout("results.txt", std::ios_base::app);
    for(int i=0; i<class_results.size(); ++i)
        fout << class_results[i] << " ";
    for(int i=0; i<class_results.size(); ++i)
        fout << reg_results[i] << " ";
    fout << std::endl;
    fout.close();


}

std::string GetOutName(const std::string& filename) {
    std::string res;
    int p = filename.find_last_of('/');
    if(p != std::string::npos) res = filename.substr(p+1);
    else res = filename;
    p = res.find_last_of('.');
    if(p != std::string::npos) res = res.substr(0,p);
    return res;
}


bool hcomparator_fs(const MeshUtils::ObjectHypothesis &h1, const MeshUtils::ObjectHypothesis &h2){
    return h1.eval.final_score > h2.eval.final_score;
}

DECLARE_string(detector_options_file);
DECLARE_string(output_folder);
void HFTest::DetectObjects() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    CHECK_GT(FLAGS_detector_options_file.size(), 0) << "You should specify a detector options file";
    std::ifstream f_dec_options(FLAGS_detector_options_file.c_str());
    CHECK(f_dec_options.is_open()) << "Detector options file not found!";

    std::stringstream sstr;
    sstr << f_dec_options.rdbuf();
    f_dec_options.close();

    DetectorOptions::Options detect_options;
    TextFormat::ParseFromString(sstr.str(), &detect_options);

    CHECK_GT(detect_options.forest_folder().size(), 0) << "No forest folder specified";
    CHECK(setInputForest(detect_options.forest_folder()));

    CHECK_GT(detect_options.caffe_definition().size(), 0) << "No caffe definition model defined.";
    CHECK_GT(detect_options.caffe_weights().size(), 0) << "No caffe weights model defined.";
    setCaffeModel(detect_options.caffe_definition(), detect_options.caffe_weights());

    CHECK_GT(detect_options.stride(), 0) << "Stride should be more than 0";
    setStrideInPixels(detect_options.stride());

    useGPU(detect_options.gpu());
    CHECK_GT(detect_options.num_threads(), 0) << "num_threads should be positive";
    setNumThreads(detect_options.num_threads());

    CHECK_GT(detect_options.max_depth_range_in_patch_in_m(), 0) << "max_depth_range_in_patch_in_m must be"
                                                                   "greater than 0 and should match the value"
                                                                   "used in training";
    setMaxDepthRange(detect_options.max_depth_range_in_patch_in_m());

    // assumes forest is already loaded
    CHECK_EQ(detect_options.object_options_size(), number_of_classes_) << "Number of objects provided in the"
                                                                          "options file does not match the "
                                                                          "number of classes in the forest";

    std::string output_folder = FLAGS_output_folder;
    if (output_folder.size() > 0 && output_folder[output_folder.size()-1] != '/')
        output_folder.push_back('/');

    setBatchSizeCaffe(detect_options.batch_size());
    setCameraIntrinsics(
                detect_options.fx(),
                detect_options.fy(),
                detect_options.cx(),
                detect_options.cy());


    MeshUtils mesh_utils;

    mesh_utils.setIntrinsics(fx_, fy_, cx_, cy_);
    mesh_utils.searchSingleObjectInstance(detect_options.search_single_object_instance());
    mesh_utils.searchSingleObjectInGroup(detect_options.search_single_object_in_group());
    mesh_utils.setNumThreads(num_threads_);

    mesh_utils.useColorSimilarity(detect_options.use_color_similarity());
    mesh_utils.setReg(detect_options.similarity_coeff(),
                      detect_options.inliers_coeff(),
                      detect_options.clutter_coeff(),
                      detect_options.location_score_coeff(),
                      detect_options.pose_score_coeff());
    mesh_utils.setGroupReg(detect_options.group_total_explain_coeff(),
                           detect_options.group_common_explain_coeff());
    mesh_utils.setInliersThreshold(detect_options.inliers_threshold());
    mesh_utils.setClutterThreshold(detect_options.clutter_threshold());
    mesh_utils.setFinalScoreThreshold(detect_options.final_score_threshold());

    mesh_utils.setClusteringOptions(detect_options.cluster_eps_angle_threshold(),
                                    detect_options.cluster_min_points(),
                                    detect_options.cluster_curvature_threshold(),
                                    detect_options.cluster_tolerance_near(),
                                    detect_options.cluster_tolerance_far());

    for(int i=0; i<detect_options.object_options_size(); i++) {
        if(detect_options.object_options(i).should_detect()) {
            const DetectorOptions::ObjectOptions& opt = detect_options.object_options(i);
            mesh_utils.insertObjectFromPLY(opt.mesh_file(), i, opt.name(), false /*opt.align_z_axis()*/,
                                           opt.nn_search_radius(), opt.icp_iterations());
        }
    }

    bool use_random_values_in_patches = !detect_options.are_objects_segmented();

    std::string rgb_fname, depth_fname;
    while(cin >> rgb_fname >> depth_fname) {


        cv::Mat rgb = cv::imread(rgb_fname);
        if(rgb.empty()) {
            cout << "Cannot read file: " << rgb_fname << std::endl;
            continue;
        }

        cv::Mat depth = cv::imread(depth_fname, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        if(depth.empty()) {
            cout << "Cannot read file: " << depth_fname << std::endl;
            continue;
        }

        cv::Mat rgb_res;
        rgb.copyTo(rgb_res);

        mesh_utils.setScene(rgb, depth, detect_options.distance_threshold());
        std::vector<MeshUtils::ObjectHypothesis> h_vec = test_image(
                    rgb, depth, detect_options, mesh_utils, use_random_values_in_patches,
                    detect_options.distance_threshold());

        std::sort(h_vec.begin() ,h_vec.end(), hcomparator_fs);
        std::vector<int> hcounter(detect_options.object_options_size(), 0);

	std::string out_name = GetOutName(rgb_fname);
        std::string out_fname = output_folder + out_name + "_res.txt";
        std::ofstream fout(out_fname.c_str());
        CHECK(fout) << "Cannot write to output file " << out_fname;
	std::cout << "Writing info to: " << out_fname << std::endl;
	int total_found = 0;
        for(int h=0; h<h_vec.size(); ++h){
            int obj_id = h_vec[h].obj_id;
            if(hcounter[obj_id] < detect_options.object_options(obj_id).instances()) {
                hcounter[obj_id]++;
		total_found++;
                fout << detect_options.object_options(obj_id).name() << "(" << hcounter[obj_id] << ")" << ": " << std::endl;
		/* for debugging
                fout << "---------" << std::endl;
                fout << "clutter score: " << h_vec[h].eval.clutter_score << std::endl;
                fout << "similarity score: " << h_vec[h].eval.similarity_score << std::endl;
                fout << "inliers ratio: " << h_vec[h].eval.inliers_ratio << std::endl;
                fout << "visibility ratio: " << h_vec[h].eval.visibility_ratio << std::endl;
                fout << "location score: " << h_vec[h].eval.location_score << std::endl;
                fout << "pose score: " << h_vec[h].eval.pose_score << std::endl;
                fout << "final score: " << h_vec[h].eval.final_score << std::endl;
		*/
		fout << h_vec[h].rotmat << std::endl;
                // TODO
                // provide the ground truth to measure the error correctly.
                // fout << "ground truth error : " << h_vec[h].eval.ground_truth_error << std::endl;
                fout << std::endl;

		//TODO
		//implement the alignment functionality
                //if(mesh_utils.is_plane_detected() &&
                // detect_options.object_options(obj_id).align_z_axis())
                //   correctPose(h_vec[h].rotmat, mesh_utils.getUpVector());

                mesh_utils.renderObject(rgb_res, h_vec[h].obj_id, h_vec[h].rotmat, 1.0f);
                cv::Scalar color;
                color = cv::Scalar(0, 255, 0);
                //mesh_utils.draw_hypotheses_boundingbox(h_tmp, rgb_res, color, 2);
                //mesh_utils.draw_hypotheses_boundingbox(best_h[h], rgb_res, color, 2);

            }
        }
	std::string rgb_out_fname = output_folder + out_name + "_res.png";
	std::cout << "Writing result image to: " << rgb_out_fname << std::endl;
        cv::imwrite(rgb_out_fname, rgb_res);

        fout.close();
	std::cout << "Detection finished. Total objects found: " << total_found << std::endl;
    }

}


