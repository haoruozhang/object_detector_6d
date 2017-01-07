/* Output File Format:
 * 1st int: Number of classes
 * 2nd int: output feature vector length N
 * Follow P (unknown) patches:
 *      int object_id,
 *      float yaw, pitch, roll
 *      float x, y, z (in object coordinates)
 *      float[N] output vector
 */

#include <iostream>

#include <train_patch_generator.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <boost/algorithm/string.hpp>

#include <cv.h>
#include <highgui.h>
#include <fstream>


void train_patch_generator::generate_train_patches(){

    //check caffe files
    CHECK_GT(caffe_model_definition_filename_.size(), 0) << "No caffe definition model defined.";
    CHECK_GT(caffe_model_weights_filename_.size(), 0) << "No caffe weights model defined.";

    //caffe::Caffe::set_phase(caffe::Caffe::TEST);  // used with previous version of Caffe
    caffe::Net<float> caffe_net(caffe_model_definition_filename_, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(caffe_model_weights_filename_);

    //create lmdb database
    MDB_env* mdb_env;
    MDB_dbi mdb_dbi;
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_val mdb_key, mdb_value;
    CHECK_GT(input_lmdb_.size(), 0) << "No lmdb input specified.";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env,
             input_lmdb_.c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    //annotation file
    std::ifstream fannot((input_lmdb_ + "/patch_annotation_lmdb.txt").c_str());
    CHECK(fannot) << "Cannot open annotation file: " << input_lmdb_ + "/patch_annotation_lmdb.txt";

    CHECK(output_file_.size() > 0) << "No output file specified";

    //output file
    std::ofstream fout(output_file_.c_str(), std::ios::out | std::ios::binary);
    CHECK(fout) << "Could not open output file " << output_file_ << " for writing.";

    //Get number of classes
    int num_classes;
    fannot >> num_classes;
    //write number of classes
    fout.write((char*)&num_classes, sizeof(int));

    bool output_vector_length_written = false;
    int num_patches = 0;
    while(true){

        bool data_end = false;
        std::vector<Annotation> annot_vec;
        std::vector<float> float_data;
        for(int b=0; b<batch_size_; ++b){

            //get patch from lmdb
            caffe::Datum datum;
            datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
            std::string key((char*)mdb_key.mv_data);
            key.resize(13); //format: xxxx_yyyyyyyy -> x: obj number, y: patch number

            //get annotation
            Annotation annot;
            CHECK(fannot >> annot.annot_key >> annot.yaw >> annot.pitch >> annot.roll >> annot.obj_x >> annot.obj_y >> annot.obj_z)
                    << "Couldn't read annotation. Maybe not enough entries?";

            CHECK_EQ(key, annot.annot_key) << "annotation and database key mismatch";

            annot_vec.push_back(annot);

            //normalize & copy patch to Net
            //std::vector<float> float_data((unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
            int start = float_data.size();
            float_data.insert(float_data.end(), (unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
            //apply same scale to the input as in training
            for(int i=start; i<float_data.size(); ++i)
                float_data[i] /= 255.0f;

            if(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) != MDB_SUCCESS){
                data_end = true;
                break;
            }

        }

        if(data_end)
            break;

        if(use_gpu_)
            caffe::caffe_copy(caffe_net.input_blobs()[0]->count(), &(float_data[0]), caffe_net.input_blobs()[0]->mutable_gpu_data());
        else
            caffe::caffe_copy(caffe_net.input_blobs()[0]->count(), &(float_data[0]), caffe_net.input_blobs()[0]->mutable_cpu_data());

        //Get feature from Net - forward image patch
        //output_blob[0]: contains the result
        //output_blob[1]: contains the input
        const std::vector<caffe::Blob<float>* > output_blob = caffe_net.ForwardPrefilled();

        for(int b=0; b<batch_size_; ++b){

            int feature_vector_length = output_blob[0]->count() / batch_size_;

            //write vector length only once
            if(!output_vector_length_written){
                int output_vector_length = feature_vector_length;
                fout.write((char*)&output_vector_length, sizeof(int));
                output_vector_length_written = true;
            }

            //write annotation & feature vector
            std::vector<std::string> object_id_str;
            boost::split(object_id_str, annot_vec[b].annot_key, boost::is_any_of("_"));
            int obj_id = boost::lexical_cast<int>(object_id_str[0].c_str(), object_id_str[0].size());

            fout.write((char*)&obj_id,  sizeof(int));
            fout.write((char*)&annot_vec[b].yaw,     sizeof(float));
            fout.write((char*)&annot_vec[b].pitch,   sizeof(float));
            fout.write((char*)&annot_vec[b].roll,    sizeof(float));
            fout.write((char*)&annot_vec[b].obj_x,   sizeof(float));
            fout.write((char*)&annot_vec[b].obj_y,   sizeof(float));
            fout.write((char*)&annot_vec[b].obj_z,   sizeof(float));

            for(int i=0; i<feature_vector_length; ++i){
                float f = output_blob[0]->data_at(b, i, 0, 0);
                fout.write((char*)&f, sizeof(float));
            }

//    //      visualize patches and output, works only with full network, i.e. full reconstruction
//            cv::Mat rgb1(patch_size_, patch_size_, CV_8UC3);
//            cv::Mat rgb2(patch_size_, patch_size_, CV_8UC3);
//            for(int c=0; c<3; ++c){
//                for(int h=0; h<patch_size_; ++h){
//                    for(int w=0; w<patch_size_; ++w){
//                        int idx = c*patch_size_*patch_size_ + h*patch_size_ + w;
//                        rgb1.at<cv::Vec3b>(h, w)[c] = static_cast<unsigned char>(output_blob[0]->data_at(b, idx, 0, 0) * 255.0f);
//                        rgb2.at<cv::Vec3b>(h, w)[c] = static_cast<unsigned char>(output_blob[1]->data_at(b, idx, 0, 0) * 255.0f);
//                    }
//                }
//            }

//            cv::resize(rgb1, rgb1, cv::Size(100, 100));
//            cv::resize(rgb2, rgb2, cv::Size(100, 100));
//            int k=-1;
//            while(k==-1){
//                cv::imshow("rgb1", rgb1);
//                cv::imshow("rgb2", rgb2);
//                k = cv::waitKey(30);
//            }

        }

        num_patches += batch_size_;
        std::cout << "Training samples written: " << num_patches << "\r";


    }
    std::cout << std::endl << "Finished!" << std::endl;

    //close lmdb database
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);

    //close files
    fannot.close();
    fout.close();

}



void train_patch_generator::generate_train_patches_pixeltests(){

    //create lmdb database
    MDB_env* mdb_env;
    MDB_dbi mdb_dbi;
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_val mdb_key, mdb_value;
    CHECK_GT(input_lmdb_.size(), 0) << "No lmdb input specified.";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env,
             input_lmdb_.c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    //annotation file
    std::ifstream fannot((input_lmdb_ + "/patch_annotation_lmdb.txt").c_str());
    CHECK(fannot) << "Cannot open annotation file: " << input_lmdb_ + "/patch_annotation_lmdb.txt";

    CHECK(output_file_.size() > 0) << "No output file specified";

    //output file
    std::ofstream fout(output_file_.c_str(), std::ios::out | std::ios::binary);
    CHECK(fout) << "Could not open output file " << output_file_ << " for writing.";

    //Get number of classes
    int num_classes;
    fannot >> num_classes;
    //write number of classes
    fout.write((char*)&num_classes, sizeof(int));

    bool output_vector_length_written = false;
    int num_patches = 0;

    int feature_vector_length = 0;

    while(true){

        bool data_end = false;
        std::vector<Annotation> annot_vec;
        std::vector<float> float_data;
        for(int b=0; b<batch_size_; ++b){

            //get patch from lmdb
            caffe::Datum datum;
            datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
            std::string key((char*)mdb_key.mv_data);
            key.resize(13); //format: xxxx_yyyyyyyy -> x: obj number, y: patch number

            //get annotation
            Annotation annot;
            CHECK(fannot >> annot.annot_key >> annot.yaw >> annot.pitch >> annot.roll >> annot.obj_x >> annot.obj_y >> annot.obj_z)
                    << "Couldn't read annotation. Maybe not enough entries?";

            CHECK_EQ(key, annot.annot_key) << "annotation and database key mismatch";

            annot_vec.push_back(annot);

            //normalize & copy patch to Net
            //std::vector<float> float_data((unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
            int start = float_data.size();
            float_data.insert(float_data.end(), (unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
            //apply same scale to the input as in training
            for(int i=start; i<float_data.size(); ++i)
                float_data[i] /= 255.0f;

            if(feature_vector_length == 0)
                feature_vector_length = datum.data().size();

            if(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) != MDB_SUCCESS){
                data_end = true;
                break;
            }

        }

        if(data_end)
            break;

        for(int b=0; b<batch_size_; ++b){

            //write vector length only once
            if(!output_vector_length_written){
                int output_vector_length = feature_vector_length;
                fout.write((char*)&output_vector_length, sizeof(int));
                output_vector_length_written = true;
            }

            //write annotation & feature vector
            std::vector<std::string> object_id_str;
            boost::split(object_id_str, annot_vec[b].annot_key, boost::is_any_of("_"));
            int obj_id = boost::lexical_cast<int>(object_id_str[0].c_str(), object_id_str[0].size());

            fout.write((char*)&obj_id,  sizeof(int));
            fout.write((char*)&annot_vec[b].yaw,     sizeof(float));
            fout.write((char*)&annot_vec[b].pitch,   sizeof(float));
            fout.write((char*)&annot_vec[b].roll,    sizeof(float));
            fout.write((char*)&annot_vec[b].obj_x,   sizeof(float));
            fout.write((char*)&annot_vec[b].obj_y,   sizeof(float));
            fout.write((char*)&annot_vec[b].obj_z,   sizeof(float));

            fout.write((char*)&float_data[b*feature_vector_length], feature_vector_length * sizeof(float));

        }

        num_patches += batch_size_;
        std::cout << "Training samples written: " << num_patches << "\r";


    }
    std::cout << std::endl << "Finished!" << std::endl;

    //close lmdb database
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);

    //close files
    fannot.close();
    fout.close();

}

void train_patch_generator::generate_train_patches_kmeans_centers(){

    //create lmdb database
    MDB_env* mdb_env;
    MDB_dbi mdb_dbi;
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_val mdb_key, mdb_value;
    CHECK_GT(input_lmdb_.size(), 0) << "No lmdb input specified.";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env,
             input_lmdb_.c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    //annotation file
    std::ifstream fannot((input_lmdb_ + "/patch_annotation_lmdb.txt").c_str());
    CHECK(fannot) << "Cannot open annotation file: " << input_lmdb_ + "/patch_annotation_lmdb.txt";

    CHECK(output_file_.size() > 0) << "No output file specified";


    //Get number of classes
    int num_classes;
    fannot >> num_classes;    


    int num_patches = 0;

    int feature_vector_length = 0;

    std::vector<float> float_data;
    std::vector<Annotation> annot_vec;

    while(true){

        bool data_end = false;

        //get patch from lmdb
        caffe::Datum datum;
        datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
        std::string key((char*)mdb_key.mv_data);
        key.resize(13); //format: xxxx_yyyyyyyy -> x: obj number, y: patch number

        //get annotation
        Annotation annot;
        CHECK(fannot >> annot.annot_key >> annot.yaw >> annot.pitch >> annot.roll >> annot.obj_x >> annot.obj_y >> annot.obj_z)
                << "Couldn't read annotation. Maybe not enough entries?";

        CHECK_EQ(key, annot.annot_key) << "annotation and database key mismatch";

        annot_vec.push_back(annot);

        //normalize & copy patch to Net
        //std::vector<float> float_data((unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
        int start = float_data.size();
        float_data.insert(float_data.end(), (unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
        //apply same scale to the input as in training
        for(int i=start; i<float_data.size(); ++i)
            float_data[i] /= 255.0f;

        if(feature_vector_length == 0)
            feature_vector_length = datum.data().size();

        if(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) != MDB_SUCCESS){
            data_end = true;
            break;
        }


        if(data_end)
            break;

    }

    int nvectors = float_data.size() / feature_vector_length;
    cv::Mat data(nvectors, feature_vector_length, CV_32FC1);
    std::cout << "nvectors: " << nvectors << std::endl;

    //nvectors = 800;
    int nfeatures = 400;

    for(int row = 0; row < nvectors; ++row)
        for(int col = 0; col < feature_vector_length; ++col)
            data.at<float>(row, col) = float_data[row*feature_vector_length + col];

    cv::Mat bestLabels;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 0.001);
    cv::Mat centers;
    cv::kmeans(data, nfeatures, bestLabels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

    cv::FileStorage fout_centers(output_file_ + ".centers.xml", cv::FileStorage::WRITE);
    fout_centers << "centers" << centers;
    fout_centers.release();




    std::cout << std::endl << "Finished!" << std::endl;

    //close lmdb database
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);

    //close files
    fannot.close();    

}


void train_patch_generator::generate_train_patches_kmeans_vectors(){

    //create lmdb database
    MDB_env* mdb_env;
    MDB_dbi mdb_dbi;
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_val mdb_key, mdb_value;
    CHECK_GT(input_lmdb_.size(), 0) << "No lmdb input specified.";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env,
             input_lmdb_.c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";

    //annotation file
    std::ifstream fannot((input_lmdb_ + "/patch_annotation_lmdb.txt").c_str());
    CHECK(fannot) << "Cannot open annotation file: " << input_lmdb_ + "/patch_annotation_lmdb.txt";

    CHECK(output_file_.size() > 0) << "No output file specified";

    //output file
    std::ofstream fout(output_file_.c_str(), std::ios::out | std::ios::binary);
    CHECK(fout) << "Could not open output file " << output_file_ << " for writing.";

    //Get number of classes
    int num_classes;
    fannot >> num_classes;
    //write number of classes
    fout.write((char*)&num_classes, sizeof(int));


    int num_patches = 0;

    int feature_vector_length = 0;

    std::vector<float> float_data;
    std::vector<Annotation> annot_vec;

    while(true){

        bool data_end = false;

        //get patch from lmdb
        caffe::Datum datum;
        datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
        std::string key((char*)mdb_key.mv_data);
        key.resize(13); //format: xxxx_yyyyyyyy -> x: obj number, y: patch number

        //get annotation
        Annotation annot;
        CHECK(fannot >> annot.annot_key >> annot.yaw >> annot.pitch >> annot.roll >> annot.obj_x >> annot.obj_y >> annot.obj_z)
                << "Couldn't read annotation. Maybe not enough entries?";

        CHECK_EQ(key, annot.annot_key) << "annotation and database key mismatch";

        annot_vec.push_back(annot);

        //normalize & copy patch to Net
        //std::vector<float> float_data((unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
        int start = float_data.size();
        float_data.insert(float_data.end(), (unsigned char*)&datum.data()[0], (unsigned char*)&datum.data()[0] + datum.data().size());
        //apply same scale to the input as in training
        for(int i=start; i<float_data.size(); ++i)
            float_data[i] /= 255.0f;

        if(feature_vector_length == 0)
            feature_vector_length = datum.data().size();

        if(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) != MDB_SUCCESS){
            data_end = true;
            break;
        }


        if(data_end)
            break;

    }

    int nvectors = float_data.size() / feature_vector_length;
    cv::Mat data(nvectors, feature_vector_length, CV_32FC1);
    std::cout << "nvectors: " << nvectors << std::endl;

    //nvectors = 800;
    int nfeatures = 400;

    for(int row = 0; row < nvectors; ++row)
        for(int col = 0; col < feature_vector_length; ++col)
            data.at<float>(row, col) = float_data[row*feature_vector_length + col];


    cv::Mat centers;
    cv::FileStorage fout_centers(output_file_ + ".centers.xml", cv::FileStorage::READ);
    fout_centers["centers"] >> centers;
    fout_centers.release();


    fout.write((char*)&nfeatures, sizeof(int));

    for(int b=0; b<nvectors; ++b){
        //write annotation & feature vector
        std::vector<std::string> object_id_str;
        boost::split(object_id_str, annot_vec[b].annot_key, boost::is_any_of("_"));
        int obj_id = boost::lexical_cast<int>(object_id_str[0].c_str(), object_id_str[0].size());

        fout.write((char*)&obj_id,  sizeof(int));
        fout.write((char*)&annot_vec[b].yaw,     sizeof(float));
        fout.write((char*)&annot_vec[b].pitch,   sizeof(float));
        fout.write((char*)&annot_vec[b].roll,    sizeof(float));
        fout.write((char*)&annot_vec[b].obj_x,   sizeof(float));
        fout.write((char*)&annot_vec[b].obj_y,   sizeof(float));
        fout.write((char*)&annot_vec[b].obj_z,   sizeof(float));

        std::vector<float> z(nfeatures);
        float meanz = 0;
        for(int k=0; k<nfeatures; ++k){
            z[k] = 0;
            for(int i=0; i<feature_vector_length; ++i)
                z[k] += pow(data.at<float>(b, i) - centers.at<float>(k, i), 2);
            z[k] = sqrt(z[k]);
            meanz += z[k] / nfeatures;
        }

        std::vector<float> fvec(nfeatures, 0);
        for(int k=0; k<nfeatures; ++k)
            fvec[k] = std::max((float)0, float(meanz - z[k]));


        fout.write((char*)&(fvec[0]), nfeatures * sizeof(float));

        num_patches++;
        std::cout << "Training samples written: " << num_patches << "\r";
    }


    std::cout << std::endl << "Finished!" << std::endl;

    //close lmdb database
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);

    //close files
    fannot.close();
    fout.close();

}
