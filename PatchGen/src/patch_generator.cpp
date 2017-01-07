/*

lmdb output:
------------
key: oooo_pppppppp (o: object id, p: patch id)
data: floats: Channel  x  voxels  x  voxels (different from bin file)


*/



#include <patch_generator.h>
#include <highgui.h>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>


//yaw pitch roll --> rads!
void patch_generator::get_yaw_pitch_roll_from_rot_mat(const Eigen::Matrix4f &rot_mat, float &yaw,
                                                      float &pitch, float &roll){

    yaw = atan2(rot_mat(1, 0), rot_mat(0, 0));
    float a = sqrt(rot_mat(2, 1)*rot_mat(2, 1) + rot_mat(2, 2)*rot_mat(2, 2));
    pitch = atan2(-rot_mat(2,0), a);
    roll = atan2(rot_mat(2,1), rot_mat(2,2));

}

Eigen::Vector4f patch_generator::get_object_coords(int im_width, int im_height, int w, int h,
                                                   unsigned short depth, const Eigen::Matrix4f &rot_mat){

    float focal_length = getFocalLength(im_width, im_height);
    float cx = (float)im_width / 2.0f - 0.5f;
    float cy = (float)im_height / 2.0f - 0.5f;
    float z = (float)depth / 1000.0f;
    float x = ((float)w - cx)*z/focal_length;
    float y = ((float)h - cy)*z/focal_length;
    Eigen::Vector4f pos;
    pos << x, y, z, 1;

    //rotate rot_mat (obj_pose) 180 degrees around X
    //to match xtion coordinate system
    Eigen::Matrix4f corrMat;
    corrMat << 1,  0,  0, 0,
               0, -1,  0, 0,
               0,  0, -1, 0,
               0,  0,  0, 1;
    Eigen::Matrix4f xtion_rot_mat = corrMat * rot_mat;

    return xtion_rot_mat.inverse() * pos;

}

// NOT USED
void patch_generator::insert_patches_to_db(const std::vector<float> &patches,
                                           caffe::Datum &datum,
                                           int cur_obj, int &patch_id,
                                           const Eigen::Matrix4f &obj_pose,
                                           const cv::Mat &depth,
                                           const std::vector<int> patches_loc,
                                           std::ofstream &fannot)
{

    //patches needs to be in Channel - Row - Col

    const int kMaxKeyLength = 20;
    char key_cstr[kMaxKeyLength];

    float yaw, pitch, roll;
    get_yaw_pitch_roll_from_rot_mat(obj_pose, yaw, pitch, roll);

    int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 6;
    for(int i=0; i<num_patches; ++i){

        datum.clear_float_data();
        datum.clear_data();
        std::string* datum_string = datum.mutable_data();
        datum_string->resize(patch_size_in_voxels_ * patch_size_in_voxels_ * 6);
        int pos=0;
        for(int c=0; c<6; ++c){
            for(int row=0; row<patch_size_in_voxels_; ++row){
                for(int col=0; col<patch_size_in_voxels_; ++col){
                    int idx = i * patch_size_in_voxels_ * patch_size_in_voxels_ * 6 +
                              row * patch_size_in_voxels_ * 6 +
                              col * 6 +
                              c;

                    //surface normals are also quantized into 256 bins.
                    if(c<3)
                        (*datum_string)[pos++] = static_cast<unsigned char>(patches[idx] * 255.0f);
                    else
                        (*datum_string)[pos++] = static_cast<unsigned char>( (patches[idx]/2.0 + 0.5f) * 255.0f);

                }
            }
        }        

        //write datum(=patch) to db
        snprintf(key_cstr, kMaxKeyLength, "%04d_%08d", cur_obj, patch_id++);
        std::string keystr(key_cstr);

        std::string value;
        datum.SerializeToString(&value);


        mdb_data.mv_size = value.size();
        mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
        mdb_key.mv_size = keystr.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
        CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
            << "mdb_put failed";

        //write annotation file
        //get patch location in depth image
        int patch_center_x = patches_loc[i*2];
        int patch_center_y = patches_loc[i*2 + 1];
        unsigned short d = depth.at<unsigned short>(patch_center_y, patch_center_x);
        Eigen::Vector4f patch_coords = get_object_coords(depth.cols, depth.rows, patch_center_x, patch_center_y, d, obj_pose);
        fannot << keystr << " " <<
                  yaw    << " " <<
                  pitch  << " " <<
                  roll   << " " <<
                  patch_coords(0) << " " <<
                  patch_coords(1) << " " <<
                  patch_coords(2) << std::endl;

    }

}

// NOT USED
void patch_generator::generatePatches(){


    CHECK(input_object_folders_.size() != 0) << "No input objects specified";
    //CHECK_EQ(mkdir(output_folder_, 0744), 0) << "mkdir " << output_folder_ << "failed";

    //for bin files
    std::ofstream fout;
    //for lmdb
    caffe::Datum datum;
    datum.clear_data();

    //write info file
    std::string finfo_str;
    if(out_type_ == OUTPUT_BIN)
        finfo_str = output_folder_ + "/patch_info_bin.txt";
    else if(out_type_ == OUTPUT_LMDB)
        finfo_str = output_folder_ + "/patch_info_lmdb.txt";

    std::ofstream finfo(finfo_str.c_str());
    CHECK(finfo) << "Cannot open file " << finfo_str << " for writing.";


    //write annotation file
    std::string fannot_str;
    if(out_type_ == OUTPUT_BIN)
        fannot_str = output_folder_ + "/patch_annotation_bin.txt";
    else if(out_type_ == OUTPUT_LMDB)
        fannot_str = output_folder_ + "/patch_annotation_lmdb.txt";

    std::ofstream fannot(fannot_str.c_str());
    CHECK(fannot) << "Cannot open file " << fannot_str << " for writing.";

    //write number of classes
    fannot << input_object_folders_.size() << std::endl;


    if(out_type_ == OUTPUT_LMDB){

        LOG(INFO) << "Opening lmdb " << output_folder_;
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS)
                << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)   // 1TB
                << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, output_folder_.c_str(), 0, 0664), MDB_SUCCESS)
                << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";

        datum.set_channels(6); //B,G,R,X,Y,Z
        datum.set_height(patch_size_in_voxels_);
        datum.set_width(patch_size_in_voxels_);

    }

    finfo << "Patch size in voxels: " << patch_size_in_voxels_ << std::endl;
    finfo << "Voxel size in m: " << voxel_size_in_m_ << std::endl;
    finfo << "Stride in pixels: " << stride_in_pixels_ << std::endl;


    int num_total_patches = 0;
    for(int cur_obj = 0; cur_obj < input_object_folders_.size(); ++cur_obj){

        int num_obj_patches = 0;
        //get object name from folder
        std::vector<std::string> folder_split;
        boost::split(folder_split, input_object_folders_[cur_obj], boost::is_any_of("/") );
        std::string cur_obj_str = folder_split.back();
        std::cout << "Creating patches for object: " << cur_obj_str << std::endl;


        //Initialize proper output - write config parameters
        if(out_type_ == OUTPUT_BIN){
            std::string cur_file = output_folder_ + "/" + cur_obj_str + ".bin";
            fout.open(cur_file.c_str(), std::ios::out | std::ios::binary);
            CHECK(fout) << "Output file " << (output_folder_ + "/" + cur_obj_str) << " cannot be openned.";
            fout.write((char*)&patch_size_in_voxels_, sizeof(int));
            fout.write((char*)&voxel_size_in_m_, sizeof(float));
        } else if(out_type_ == OUTPUT_LMDB) {
            datum.set_label(cur_obj);
        }

        int file_counter = 0;
        int patch_id = 0;
        while(true){

//            if(file_counter > 100)
//                break;

            //read files
            std::stringstream fname;
            fname << input_object_folders_[cur_obj] << "/rgb" << file_counter << ".png";
            cv::Mat rgb = cv::imread(fname.str());
            if(rgb.empty())
                break;
            //read depth
            fname.str("");
            fname << input_object_folders_[cur_obj] << "/depth" << file_counter << ".png";
            cv::Mat depth = cv::imread(fname.str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            CHECK(!depth.empty()) << "File " << fname.str() << " not exist, while the rgb file does.";            
            //read normals
            fname.str("");
            fname << input_object_folders_[cur_obj] << "/surface_normals" << file_counter << ".bin";
            std::ifstream fnormals(fname.str().c_str(), std::ios::out | std::ios::binary);
            CHECK(fnormals) << "File " << fname.str() << " not exist, while the rgb and depth files does.";
            std::vector<float> normals(rgb.rows * rgb.cols * 3);
            fnormals.read(reinterpret_cast<char*>(&normals[0]), normals.size() * sizeof(float));
            //read pose
            fname.str("");
            fname << input_object_folders_[cur_obj] << "/pose" << file_counter << ".txt";
            std::ifstream fpose(fname.str().c_str());
            CHECK(fpose) << "File " << fname.str() << " not exist, while the rgb and depth files does.";
            Eigen::Matrix4f obj_pose;
            for(int row=0; row<4; ++row){
                for(int col=0; col<4; ++col){
                    float a;
                    fpose >> a;
                    obj_pose(row, col) = a;
                }
            }



            //create 3D texture vector Rows x Cols x 7 (B, G, R, D, (X, Y, Z)normals )
            std::vector<float> texture_3D(rgb.rows * rgb.cols * 7);
            int tex_pos = 0;
            int norm_pos = 0;
            for(int row=0; row<rgb.rows; ++row ){
                for(int col=0; col<rgb.cols; ++col){
                    texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[0] / 255.0f;  //B
                    texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[1] / 255.0f;  //G
                    texture_3D[tex_pos++] = (float)rgb.at<cv::Vec3b>(row, col)[2] / 255.0f;  //R
                    texture_3D[tex_pos++] = (float)depth.at<unsigned short>(row, col);       //D (in mm)
                    texture_3D[tex_pos++] = normals[norm_pos++];                             //X norm
                    texture_3D[tex_pos++] = normals[norm_pos++];                             //Y norm
                    texture_3D[tex_pos++] = normals[norm_pos++];                             //Z norm
                }
            }            


            std::vector<float> patches;
            //patch locations - [x1, y1, x2, y2 .. ]
            std::vector<int> patches_loc;
            patch_extractor_gpu::extract_patches(texture_3D,
                                                 rgb.cols,
                                                 rgb.rows,
                                                 patch_size_in_voxels_,
                                                 voxel_size_in_m_,
                                                 stride_in_pixels_,
                                                 getFocalLength(rgb.cols, rgb.rows),
                                                 patches,
                                                 patches_loc,
                                                 generate_random_values_,
                                                 distance_threshold_);


            if(out_type_ == OUTPUT_BIN){
                fout.write((char*)&patches[0], patches.size() * sizeof(float));
            } else if(out_type_ == OUTPUT_LMDB){
                insert_patches_to_db(patches, datum, cur_obj, patch_id, obj_pose, depth, patches_loc, fannot);
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_commit failed";
                CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_begin failed";

            }

            int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 6;
            num_obj_patches += num_patches;

            file_counter++;
            std::cout << "Extracted patches from file: " << file_counter << "\r";


// visualize patches:
//
//            //int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 6;
//            int patches_pos = 0;
//            for(int i=0; i<num_patches; ++i){

//                cv::Mat patch_rgb = cv::Mat(patch_size_in_voxels_, patch_size_in_voxels_, CV_8UC3);
//                bool is_empty = true;
//                for(int row = 0; row<patch_size_in_voxels_; ++row){
//                    for(int col=0; col<patch_size_in_voxels_; ++col){

//                        patch_rgb.at<cv::Vec3b>(row,col)[0] = (uchar)(patches[patches_pos++]*255.0f);
//                        patch_rgb.at<cv::Vec3b>(row,col)[1] = (uchar)(patches[patches_pos++]*255.0f);
//                        patch_rgb.at<cv::Vec3b>(row,col)[2] = (uchar)(patches[patches_pos++]*255.0f);

//                        float x = patches[patches_pos++];
//                        float y = patches[patches_pos++];
//                        float z = patches[patches_pos++];

//                        if(x!=0 || y!=0 || z!= 0)
//                            is_empty = false;

//                    }
//                }

//                if(!is_empty){
//                    int k = -1;
//                    while(k == -1){
//                        cv::imshow("patch", patch_rgb);
//                        k = cv::waitKey(30);
//                    }
//                }
//                else{
//                    std::cout << "Got empty patch!" << std::endl;
//                }

//            }//patch iterator
// //////////////////////////////

        } // file iterator

        if(out_type_ == OUTPUT_BIN){
            fout.close();
        }        

        num_total_patches += num_obj_patches;
        std::cout << std::endl;
        std::cout << "Patches for " << cur_obj_str << ": " << num_obj_patches << std::endl;
        finfo << "Patches for " << cur_obj_str << ": " << num_obj_patches << std::endl;
    } //object iterator

    if(out_type_ == OUTPUT_LMDB){
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }

    std::cout << "Finished! Total patches: " << num_total_patches << std::endl;
    finfo << "Total patches: " << num_total_patches << std::endl;
    finfo.close();
    fannot.close();
} //generatePatches










void patch_generator::insert_patches_to_db_rgbd(const std::vector<float> &patches,
                                           caffe::Datum &datum,
                                           int cur_obj, int &patch_id,
                                           const Eigen::Matrix4f &obj_pose,
                                           const cv::Mat &depth,
                                           const std::vector<int> patches_loc,
                                           std::ofstream &fannot)
{

    //patches needs to be in Channel - Row - Col

    const int kMaxKeyLength = 20;
    char key_cstr[kMaxKeyLength];

    float yaw, pitch, roll;
    get_yaw_pitch_roll_from_rot_mat(obj_pose, yaw, pitch, roll);

    int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 4;
    for(int i=0; i<num_patches; ++i){

//        datum.clear_float_data();
//        datum.clear_data();
//        std::string* datum_string = datum.mutable_data();
//        datum_string->resize(patch_size_in_voxels_ * patch_size_in_voxels_ * 4);
//        int pos=0;
//        for(int c=0; c<4; ++c){
//            for(int row=0; row<patch_size_in_voxels_; ++row){
//                for(int col=0; col<patch_size_in_voxels_; ++col){
//                    int idx = i * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
//                              row * patch_size_in_voxels_ * 4 +
//                              col * 4 +
//                              c;

//                    (*datum_string)[pos++] = static_cast<unsigned char>(patches[idx] * 255.0f);

//                }
//            }
//        }

        //get mean
        std::vector<float> buf(patch_size_in_voxels_ * patch_size_in_voxels_ * 4);
        int pos=0;
        float mean_rgb = 0;
        float mean_depth = 0;
        for(int c=0; c<4; ++c){
            for(int row=0; row<patch_size_in_voxels_; ++row){
                for(int col=0; col<patch_size_in_voxels_; ++col){
                    int idx = i * patch_size_in_voxels_ * patch_size_in_voxels_ * 4 +
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
        }


        datum.clear_float_data();
        datum.clear_data();
        std::string* datum_string = datum.mutable_data();
        datum_string->resize(patch_size_in_voxels_ * patch_size_in_voxels_ * 4);
        //convert to char for caffe
        for(int j=0; j<buf.size(); ++j)
            (*datum_string)[j] = static_cast<unsigned char>(buf[j] * 255.0f);


        //write datum(=patch) to db
        snprintf(key_cstr, kMaxKeyLength, "%04d_%08d", cur_obj, patch_id++);
        std::string keystr(key_cstr);

        std::string value;
        datum.SerializeToString(&value);


        mdb_data.mv_size = value.size();
        mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
        mdb_key.mv_size = keystr.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
        CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
            << "mdb_put failed";

        //write annotation file
        //get patch location in depth image
        int patch_center_x = patches_loc[i*2];
        int patch_center_y = patches_loc[i*2 + 1];
        unsigned short d = depth.at<unsigned short>(patch_center_y, patch_center_x);
        Eigen::Vector4f patch_coords = get_object_coords(depth.cols, depth.rows, patch_center_x, patch_center_y, d, obj_pose);
        fannot << keystr << " " <<
                  yaw    << " " <<
                  pitch  << " " <<
                  roll   << " " <<
                  patch_coords(0) << " " <<
                  patch_coords(1) << " " <<
                  patch_coords(2) << std::endl;

    }

}







void patch_generator::generatePatches_rgbd(){


    CHECK(input_object_folders_.size() != 0) << "No input objects specified";
    //CHECK_EQ(mkdir(output_folder_, 0744), 0) << "mkdir " << output_folder_ << "failed";

    //for bin files
    std::ofstream fout;
    //for lmdb
    caffe::Datum datum;
    datum.clear_data();

    //write info file
    std::string finfo_str;
    if(out_type_ == OUTPUT_BIN)
        finfo_str = output_folder_ + "/patch_info_bin.txt";
    else if(out_type_ == OUTPUT_LMDB)
        finfo_str = output_folder_ + "/patch_info_lmdb.txt";

    std::ofstream finfo(finfo_str.c_str());
    CHECK(finfo) << "Cannot open file " << finfo_str << " for writing.";


    //write annotation file
    std::string fannot_str;
    if(out_type_ == OUTPUT_BIN)
        fannot_str = output_folder_ + "/patch_annotation_bin.txt";
    else if(out_type_ == OUTPUT_LMDB)
        fannot_str = output_folder_ + "/patch_annotation_lmdb.txt";

    std::ofstream fannot(fannot_str.c_str());
    CHECK(fannot) << "Cannot open file " << fannot_str << " for writing.";

    //write number of classes
    fannot << input_object_folders_.size() << std::endl;


    if(out_type_ == OUTPUT_LMDB){

        LOG(INFO) << "Opening lmdb " << output_folder_;
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS)
                << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)   // 1TB
                << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, output_folder_.c_str(), 0, 0664), MDB_SUCCESS)
                << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";

        datum.set_channels(4); //B,G,R,D_trunc
        datum.set_height(patch_size_in_voxels_);
        datum.set_width(patch_size_in_voxels_);

    }

    finfo << "Patch size in voxels: " << patch_size_in_voxels_ << std::endl;
    finfo << "Voxel size in m: " << voxel_size_in_m_ << std::endl;
    finfo << "Stride in pixels: " << stride_in_pixels_ << std::endl;
    finfo << "Max Depth Range: " << max_depth_range_in_m_ << std::endl;


    int num_total_patches = 0;
    for(int cur_obj = 0; cur_obj < input_object_folders_.size(); ++cur_obj){

        int num_obj_patches = 0;
        //get object name from folder
        std::vector<std::string> folder_split;
        boost::split(folder_split, input_object_folders_[cur_obj], boost::is_any_of("/") );
        std::string cur_obj_str = folder_split.back();
        std::cout << "Creating patches for object: " << cur_obj_str << std::endl;


        //Initialize proper output - write config parameters
        if(out_type_ == OUTPUT_BIN){
            std::string cur_file = output_folder_ + "/" + cur_obj_str + ".bin";
            fout.open(cur_file.c_str(), std::ios::out | std::ios::binary);
            CHECK(fout) << "Output file " << (output_folder_ + "/" + cur_obj_str) << " cannot be openned.";
            fout.write((char*)&patch_size_in_voxels_, sizeof(int));
            fout.write((char*)&voxel_size_in_m_, sizeof(float));
            fout.write((char*)&max_depth_range_in_m_, sizeof(float));
        } else if(out_type_ == OUTPUT_LMDB) {
            datum.set_label(cur_obj);
        }

        int file_counter = 0;
        int patch_id = 0;
        while(true){

//            if(file_counter > 100)
//                break;

            //read files
            std::stringstream fname;
            fname << input_object_folders_[cur_obj] << "/rgb" << file_counter << ".png";
            cv::Mat rgb = cv::imread(fname.str());
            if(rgb.empty())
                break;
            //read depth
            fname.str("");
            fname << input_object_folders_[cur_obj] << "/depth" << file_counter << ".png";
            cv::Mat depth = cv::imread(fname.str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            CHECK(!depth.empty()) << "File " << fname.str() << " not exist, while the rgb file does.";
            //read pose
            fname.str("");
            fname << input_object_folders_[cur_obj] << "/pose" << file_counter << ".txt";
            std::ifstream fpose(fname.str().c_str());
            CHECK(fpose) << "File " << fname.str() << " not exist, while the rgb and depth files does.";
            Eigen::Matrix4f obj_pose;
            for(int row=0; row<4; ++row){
                for(int col=0; col<4; ++col){
                    float a;
                    fpose >> a;
                    obj_pose(row, col) = a;
                }
            }



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
                                                 percent_,
                                                 getFocalLength(rgb.cols, rgb.rows),
                                                 patches,
                                                 patches_loc,
                                                 generate_random_values_,
                                                 distance_threshold_);


            if(out_type_ == OUTPUT_BIN){
                fout.write((char*)&patches[0], patches.size() * sizeof(float));
            } else if(out_type_ == OUTPUT_LMDB){
                insert_patches_to_db_rgbd(patches, datum, cur_obj, patch_id, obj_pose, depth, patches_loc, fannot);
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_commit failed";
                CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_begin failed";

            }

            int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 4;
            num_obj_patches += num_patches;

            file_counter++;
            std::cout << "Extracted patches from file: " << file_counter << "\r";


// visualize patches:
//
            //int num_patches = patches.size() / patch_size_in_voxels_ / patch_size_in_voxels_ / 4;
//            int patches_pos = 0;
//            for(int i=0; i<num_patches; ++i){

//                cv::Mat patch_rgb = cv::Mat(patch_size_in_voxels_, patch_size_in_voxels_, CV_8UC3);
//                cv::Mat patch_depth = cv::Mat(patch_size_in_voxels_, patch_size_in_voxels_, CV_32FC1);
//                for(int row = 0; row<patch_size_in_voxels_; ++row){
//                    for(int col=0; col<patch_size_in_voxels_; ++col){

//                        patch_rgb.at<cv::Vec3b>(row,col)[0] = (uchar)(patches[patches_pos++]*255.0f);
//                        patch_rgb.at<cv::Vec3b>(row,col)[1] = (uchar)(patches[patches_pos++]*255.0f);
//                        patch_rgb.at<cv::Vec3b>(row,col)[2] = (uchar)(patches[patches_pos++]*255.0f);
//                        patch_depth.at<float>(row, col) = patches[patches_pos++];

//                    }
//                }

//                cv::resize(patch_rgb, patch_rgb, cv::Size(100, 100));
//                cv::resize(patch_depth, patch_depth, cv::Size(100, 100));

//                int k = -1;
//                while(k == -1){
//                    cv::imshow("patch_rgb", patch_rgb);
//                    cv::imshow("patch_depth", patch_depth);
//                    k = cv::waitKey(30);
//                }


//            }//patch iterator
// //////////////////////////

        } // file iterator

        if(out_type_ == OUTPUT_BIN){
            fout.close();
        }

        num_total_patches += num_obj_patches;
        std::cout << std::endl;
        std::cout << "Patches for " << cur_obj_str << ": " << num_obj_patches << std::endl;
        finfo << "Patches for " << cur_obj_str << ": " << num_obj_patches << std::endl;
    } //object iterator

    if(out_type_ == OUTPUT_LMDB){
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }

    std::cout << "Finished! Total patches: " << num_total_patches << std::endl;
    finfo << "Total patches: " << num_total_patches << std::endl;
    finfo.close();
    fannot.close();
} //generatePatches
