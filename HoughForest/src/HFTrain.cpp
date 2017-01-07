#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <glog/logging.h>
#include <omp.h>
#include <float.h>
#include <math.h>

#include <HFTrain.h>

#include <boost/math/constants/constants.hpp>

void HFTrain::getTrainSet(TrainSet &train_set, int &number_of_classes, int &feature_vector_length)
{			    

    std::ifstream finput(input_file_.c_str(), std::ios::in | std::ios::binary);
    CHECK(finput) << "Could not open file " << input_file_;
	
    //read number of classes and feature vector length
    finput.read((char*)&number_of_classes, sizeof(int));
    finput.read((char*)&feature_vector_length, sizeof(int));

    train_set.resize(0);    

    int num_vec = 0;
    while(true){

        int class_no;
        float yaw, pitch, roll, x, y, z;

        finput.read((char*)&class_no,   sizeof(int));
        if(finput.eof())
            break;

        finput.read((char*)&yaw,        sizeof(float));
        finput.read((char*)&pitch,      sizeof(float));
        finput.read((char*)&roll,       sizeof(float));
        finput.read((char*)&x,          sizeof(float));
        finput.read((char*)&y,          sizeof(float));
        finput.read((char*)&z,          sizeof(float));

        train_set.push_back(new TrainSample());
        train_set.back()->class_no = class_no;
        train_set.back()->node = 0;
        Eigen::VectorXf dof(6);
        dof(0) = yaw;
        dof(1) = pitch;
        dof(2) = roll;
        dof(3) = x;
        dof(4) = y;
        dof(5) = z;
        train_set.back()->dof = dof;

        Eigen::VectorXf feature_vector(feature_vector_length);
        for(int i=0; i<feature_vector_length; ++i){
            float f;
            finput.read((char*)&f, sizeof(float));
            feature_vector(i) = f;
        }

        train_set.back()->feature_vector = feature_vector;

    }

    finput.close();
}



//get random train_samples from training set and put them to the front
void HFTrain::suffle_training_set(TrainSet &train_set, int train_samples, std::vector<int> &samples_per_class){

    samples_per_class.resize(number_of_classes_);
    for(int i=0; i<train_samples; ++i){
        int k = rand() % (train_set.size() - i) + i;
        TrainSample *ts;
        ts = train_set[i];
        train_set[i] = train_set[k];
        train_set[k] = ts;
        samples_per_class[train_set[i]->class_no]++;
    }
}


//takes input last leaf_id+1, returns the changed value, added the current leaves
void HFTrain::make_leafs(TrainSet &train_set, int train_samples, const std::vector<int> &samples_per_class, int &leaf_id){


    float degrees_bin_size = 15.0f;
    float coordinates_bin_size = 0.03; //1cm

    float pi = boost::math::constants::pi<float>();

    boost::unordered_map<TreeNode*, std::vector<int> > node_class_samples;
    boost::unordered_map<TreeNode*, std::vector< std::vector<Eigen::VectorXf> > > node_class_hough_votes;
//    boost::unordered_map<TreeNode*, std::vector<Hough3DMap> > node_class_houghmap;
//    boost::unordered_map<TreeNode*, std::vector<std::vector<float> > > node_class_mean;
//    boost::unordered_map<TreeNode*, std::vector<std::vector<float> > > node_class_std;

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {
        boost::unordered_map<TreeNode*, std::vector<int> > node_class_samples_local;
        boost::unordered_map<TreeNode*, std::vector< std::vector<Eigen::VectorXf> > > node_class_hough_votes_local;
//        boost::unordered_map<TreeNode*, std::vector<Hough3DMap> > node_class_houghmap_local;
//        boost::unordered_map<TreeNode*, std::vector<std::vector<float> > > node_class_mean_local;

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            if(train_set[i]->node->leaf){
                //classification info
                TreeNode *cur_node = train_set[i]->node;
                if(node_class_samples_local.count(cur_node)){
                    node_class_samples_local[cur_node][train_set[i]->class_no]++;
                } else {
                    node_class_samples_local[cur_node] = std::vector<int>(number_of_classes_, 0);
                    node_class_samples_local[cur_node][train_set[i]->class_no]++;
                }
                if(!node_class_hough_votes_local.count(cur_node))
                    node_class_hough_votes_local[cur_node] = std::vector<std::vector<Eigen::VectorXf> >(number_of_classes_);

                node_class_hough_votes_local[cur_node][train_set[i]->class_no].push_back(train_set[i]->dof);

            }
        }

        #pragma omp critical
        {
            //concatenate classification info
            for(boost::unordered_map<TreeNode*, std::vector<int> >::iterator it = node_class_samples_local.begin();
                it != node_class_samples_local.end(); ++it){
                if(node_class_samples.count(it->first)){
                    for(int i=0; i<number_of_classes_; ++i)
                        node_class_samples[it->first][i] += (it->second)[i];
                } else {
                    node_class_samples[it->first] = it->second;
                }

            }

            for(boost::unordered_map<TreeNode*, std::vector< std::vector<Eigen::VectorXf> > >::iterator it = node_class_hough_votes_local.begin();
                it != node_class_hough_votes_local.end(); ++it){

                TreeNode *cur_node = it->first;
                if(!node_class_hough_votes.count(cur_node))
                    node_class_hough_votes[cur_node] = std::vector<std::vector<Eigen::VectorXf> >(number_of_classes_);
                for(int c=0; c<number_of_classes_; ++c)
                    node_class_hough_votes[cur_node][c].insert(node_class_hough_votes[cur_node][c].end(),
                                                               node_class_hough_votes_local[cur_node][c].begin(),
                                                               node_class_hough_votes_local[cur_node][c].end());


            }

        } //critical

    } //parallel

    //normalize class probabilities and calc mean
    for(boost::unordered_map<TreeNode*, std::vector<int> >::iterator it = node_class_samples.begin();
        it != node_class_samples.end(); ++it){

        TreeNode *cur_node = it->first;
        //save id and increase id counter
        cur_node->leaf_id = leaf_id++;
        std::vector<int> *cur_class_samples = &(node_class_samples[cur_node]);
        int total_samples = 0;
        for(int i=0; i<number_of_classes_; ++i)
            total_samples += (*cur_class_samples)[i];

        //normalize class distribution according to
        //initial number of samples per class
        cur_node->class_prob.resize(number_of_classes_);
        for(int c=0; c<number_of_classes_; ++c){
            float norm = 0;
            for(int i=0; i<number_of_classes_; ++i)
                norm += (float)samples_per_class[c]/(float)samples_per_class[i]*(float)(*cur_class_samples)[i];
            cur_node->class_prob[c] = (float)(*cur_class_samples)[c] / norm ;
        }

//        //get mean by division with total samples
//        std::vector<std::vector<float> > *cur_node_class_mean = &(node_class_mean[cur_node]);
//        for(int c=0; c<number_of_classes_; ++c){
//            int cur_samples = node_class_samples[cur_node][c];
//            if(cur_samples > 0){
//                for(int f=0; f<feature_vector_length_; ++f)
//                        (*cur_node_class_mean)[c][f] /= (float)cur_samples;
//            }
//        }
    }

    for(boost::unordered_map<TreeNode*, std::vector< std::vector<Eigen::VectorXf> > >::iterator it = node_class_hough_votes.begin();
        it != node_class_hough_votes.end(); ++it){

        TreeNode *cur_node = it->first;
        cur_node->hough_votes.resize(number_of_classes_);
        for(int c=0; c<number_of_classes_; ++c)
            cur_node->hough_votes[c].insert(cur_node->hough_votes[c].end(),
                                            node_class_hough_votes[cur_node][c].begin(),
                                            node_class_hough_votes[cur_node][c].end());


    }

}


//put non-used training samples at the end and reduce train_samples accordingly
void HFTrain::clean_trainset(TrainSet& train_set, int &train_samples){
    int front = 0;
    int back = train_samples - 1;
    while(front < back){
        while(front < train_samples && !train_set[front]->node->leaf)
            front++;
        while(back >= 0 && train_set[back]->node->leaf)
            back--;
        if(front < back){
            TrainSample *t;
            t = train_set[front];
            train_set[front] = train_set[back];
            train_set[back] = t;
        }
    }
    train_samples = front;
}


//get random features, set threshold to 0 for now
void HFTrain::get_random_features(const std::vector<TreeNode*> &level_nodes,
                                  boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests
                                  )
{


    int num_tests = tests_per_node_* thresholds_per_test_;
    for(int i=0; i<level_nodes.size(); ++i)
        node_tests[level_nodes[i]].resize(num_tests);

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel for schedule(dynamic)
    for(int n=0; n<level_nodes.size(); ++n){
        TreeNode *cur_node = level_nodes[n];
        std::vector<Test> *cur_node_tests = &(node_tests[cur_node]);
        for(int t=0; t<tests_per_node_; ++t){
            int mm = rand() % 2;
            int f1 = rand() % feature_vector_length_;
            int f2 = rand() % feature_vector_length_;            
            for(int th=0; th<thresholds_per_test_; ++th){
                (*cur_node_tests)[t*thresholds_per_test_ + th].measure_mode = mm;
                (*cur_node_tests)[t*thresholds_per_test_ + th].feature1 = f1;
                (*cur_node_tests)[t*thresholds_per_test_ + th].feature2 = f2;
                (*cur_node_tests)[t*thresholds_per_test_ + th].threshold = 0;
            }
        }
    }

//    std::cout << "random tests: " << omp_get_wtime() - start << std::endl;

}


//get min-max for each test for each node to constraint thresholds' range
//also get samples per class per node and samples per node because code here is convenient
void HFTrain::get_min_max_count_samples(const TrainSet &train_set,
                          int train_samples,
                          const std::vector<TreeNode*> &level_nodes,
                          boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                          boost::unordered_map<TreeNode*, std::vector<float> > &node_test_min,
                          boost::unordered_map<TreeNode*, std::vector<float> > &node_test_max,
                          boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                          boost::unordered_map<TreeNode*, int> &node_samples
                          )
{

    int num_tests = tests_per_node_ * thresholds_per_test_;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_min[level_nodes[i]] = std::vector<float>(num_tests,  FLT_MAX);
        node_test_max[level_nodes[i]] = std::vector<float>(num_tests, -FLT_MAX);
    }

    for(int i=0; i<level_nodes.size(); ++i){
        node_class_samples[level_nodes[i]] = std::vector<int>(number_of_classes_, 0);
        node_samples[level_nodes[i]] = 0;
    }

//    double start = omp_get_wtime();

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector<float> > node_test_min_local;
        boost::unordered_map<TreeNode*, std::vector<float> > node_test_max_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_min_local[level_nodes[i]] = std::vector<float>(num_tests,  FLT_MAX);
            node_test_max_local[level_nodes[i]] = std::vector<float>(num_tests, -FLT_MAX);
        }
        boost::unordered_map<TreeNode*, std::vector<int> > node_class_samples_local;
        boost::unordered_map<TreeNode*, int> node_samples_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_class_samples_local[level_nodes[i]] = std::vector<int>(number_of_classes_, 0);
            node_samples_local[level_nodes[i]] = 0;
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode* cur_node = train_set[i]->node;
            std::vector<float> *min_local = &node_test_min_local[cur_node];
            std::vector<float> *max_local = &node_test_max_local[cur_node];
            std::vector<Test> *cur_node_tests = &node_tests[cur_node];
            for(int t=0; t<num_tests; ++t){
                int f1 = (*cur_node_tests)[t].feature1;
                int f2 = (*cur_node_tests)[t].feature2;
                int mm = (*cur_node_tests)[t].measure_mode;
                float val;
                if(mm == 0)
                    val = train_set[i]->feature_vector(f1) - train_set[i]->feature_vector(f2);
                else if(mm == 1)
                    val = train_set[i]->feature_vector(f1);

                if(val > (*max_local)[t])
                    (*max_local)[t] = val;
                if(val < (*min_local)[t])
                    (*min_local)[t] = val;
            }
            node_class_samples_local[cur_node][train_set[i]->class_no]++;
            node_samples_local[cur_node]++;
        }
        #pragma omp critical
        {
            for(int n=0; n<level_nodes.size(); ++n){
                std::vector<float> *min_local = &node_test_min_local[level_nodes[n]];
                std::vector<float> *max_local = &node_test_max_local[level_nodes[n]];
                std::vector<float> *min_global = &node_test_min[level_nodes[n]];
                std::vector<float> *max_global = &node_test_max[level_nodes[n]];
                for(int t=0; t<num_tests; ++t){
                    if( (*min_global)[t] > (*min_local)[t] )
                        (*min_global)[t] = (*min_local)[t];

                    if( (*max_global)[t] < (*max_local)[t])
                        (*max_global)[t] = (*max_local)[t];

                }
                for(int c=0; c<number_of_classes_; ++c)
                    node_class_samples[level_nodes[n]][c] += node_class_samples_local[level_nodes[n]][c];

                node_samples[level_nodes[n]] += node_samples_local[level_nodes[n]];
            }
        }
    }


//    std::cout << "min - max: " << omp_get_wtime() - start << std::endl;
//    std::cout << node_min[level_nodes[0]][0] << " " << node_max[level_nodes[0]][0] << std::endl;

}



//get random tests
void HFTrain::get_random_thresholds(const std::vector<TreeNode*> &level_nodes,
                               boost::unordered_map<TreeNode*, std::vector<float> > &node_test_min,
                               boost::unordered_map<TreeNode*, std::vector<float> > &node_test_max,
                               boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests
                               )
{


    int num_tests = tests_per_node_* thresholds_per_test_;

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel for schedule(dynamic)
    for(int n=0; n<level_nodes.size(); ++n){
        TreeNode *cur_node = level_nodes[n];
        std::vector<Test> *cur_node_tests = &(node_tests[cur_node]);
        std::vector<float> *cur_node_test_min = &(node_test_min[cur_node]);
        std::vector<float> *cur_node_test_max = &(node_test_max[cur_node]);
        for(int t=0; t<num_tests; ++t){
            float range = (*cur_node_test_max)[t] - (*cur_node_test_min)[t];
            (*cur_node_tests)[t].threshold = (float)rand() / (float)RAND_MAX * range + (*cur_node_test_min)[t];
        }
    }

//    std::cout << "random tests: " << omp_get_wtime() - start << std::endl;

}




// ------classification------ //
//if no split found a node is marked as leaf.
//best tests are stored to the node of each training sample
void HFTrain::find_classification_split(TrainSet &train_set,
                                        int train_samples,
                                        const std::vector<TreeNode*> &level_nodes,
                                        boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                                        boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                                        boost::unordered_map<TreeNode*, int> &node_samples
                                        )
{

    //number of samples per class per test per node that goes to the left child
    //if we apply test
    int num_tests = tests_per_node_* thresholds_per_test_;
    boost::unordered_map<TreeNode*, std::vector< std::vector<int> > > node_test_class_samples_left;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_class_samples_left[level_nodes[i]].resize(num_tests);
        for(int t=0; t<num_tests; ++t)
            node_test_class_samples_left[level_nodes[i]][t] = std::vector<int>(number_of_classes_, 0);
    }

//    double start = omp_get_wtime();

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector< std::vector<int> > > node_test_class_samples_left_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_class_samples_left_local[level_nodes[i]].resize(num_tests);
            for(int t=0; t<num_tests; ++t)
                node_test_class_samples_left_local[level_nodes[i]][t] = std::vector<int>(number_of_classes_, 0);
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode *cur_node = train_set[i]->node;
            std::vector< std::vector<int> > *cur_left_local = &node_test_class_samples_left_local[cur_node];
            //apply all tests to current sample
            for(int t=0; t<num_tests; ++t){
                Test cur_test = node_tests[cur_node][t];
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                if( val < cur_test.threshold )
                    //go to left, count it
                    (*cur_left_local)[t][train_set[i]->class_no] ++;
            }

        }
        #pragma omp critical
        {

            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector<int> > *cur_left_local = &node_test_class_samples_left_local[level_nodes[n]];
                std::vector< std::vector<int> > *cur_left_global = &node_test_class_samples_left[level_nodes[n]];
                for(int t=0; t<num_tests; ++t)
                    for(int c=0; c<number_of_classes_; ++c)
                        (*cur_left_global)[t][c] += (*cur_left_local)[t][c];
            }

        }

    }

//    std::cout << "class hist: " << omp_get_wtime() - start << std::endl;

    //compute best test via entropy

#pragma omp parallel
    {
    int num_leaves = 0;
    #pragma omp for schedule(dynamic)
    for(int n=0; n<level_nodes.size(); ++n){
        TreeNode *cur_node = level_nodes[n];
        std::vector< std::vector<int> > *cur_samples_left = &node_test_class_samples_left[cur_node];
        std::vector<int> *cur_class_samples = &node_class_samples[cur_node];
        int cur_samples = node_samples[cur_node];
        float min_entropy = FLT_MAX;
        Test best_test;
        bool found = false;
        for(int t=0; t<num_tests; ++t){
            float entropy_left = 0;
            float entropy_right = 0;
            float entropy = FLT_MAX;
            int total_samples_left = 0;
            for(int c=0; c<number_of_classes_; ++c)
                total_samples_left += (*cur_samples_left)[t][c];
            int total_samples_right = cur_samples - total_samples_left;
            if(total_samples_left != 0 && total_samples_right != 0){
                for(int c=0; c<number_of_classes_; ++c){
                    //add left entropy
                    float p=(float)(*cur_samples_left)[t][c] / (float)total_samples_left;
                    if(p!=0) entropy_left -= p * log(p);
                    //add right entropy
                    p = (float)((*cur_class_samples)[c] - (*cur_samples_left)[t][c]) / (float)total_samples_right;
                    if(p!=0) entropy_right -= p * log(p);
                }
                entropy = (entropy_left * (float)total_samples_left + entropy_right * (float)total_samples_right);
                if(entropy < min_entropy){
                    min_entropy = entropy;
                    best_test = node_tests[cur_node][t];
                    found = true;
                }
            }

        }
        if(found){
            cur_node->test = best_test;
            cur_node->leaf = false;
            cur_node->left = new TreeNode;
            cur_node->right = new TreeNode;
        }
        else{
            cur_node->leaf = true;            
        }
    }

    }

//    std::cout << "best tests classification: " << omp_get_wtime() - start << std::endl;

}




// ------Regression for x y z------ //
//if no split found a node is marked as leaf.
//best tests are stored to the node of each training sample
void HFTrain::find_regression_location_split(TrainSet &train_set,
                                            int train_samples,
                                            const std::vector<TreeNode*> &level_nodes,
                                            boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                                            boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                                            boost::unordered_map<TreeNode*, int> &node_samples
                                            )
{

//    double start = omp_get_wtime();

    //get the mean values for all possible split-childs

    //[node][test][0-1(left-right)][0-2(x,y,z)] = mean(x, y, z) and 4->samples_of_child per child node per test per node
    int num_tests = tests_per_node_* thresholds_per_test_;
    boost::unordered_map<TreeNode*, std::vector< std::vector< std::vector<float> > > > node_test_child_mean;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_child_mean[level_nodes[i]].resize(num_tests);
        for(int t=0; t<num_tests; ++t){
            node_test_child_mean[level_nodes[i]][t].resize(2);
            for(int child=0; child<2; ++child){
                node_test_child_mean[level_nodes[i]][t][child] = std::vector<float>(4, 0); //x y z
            }
        }
    }

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector< std::vector< std::vector<float> > > > node_test_child_mean_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_child_mean_local[level_nodes[i]].resize(num_tests);
            for(int t=0; t<num_tests; ++t){
                node_test_child_mean_local[level_nodes[i]][t].resize(2);
                for(int child=0; child<2; ++child){
                    node_test_child_mean_local[level_nodes[i]][t][child] = std::vector<float>(4, 0); //x y z child_samples
                }
            }
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode *cur_node = train_set[i]->node;
            //reduce calls to hash table so to reduce execution time
            std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean_local[cur_node]);
            //apply all tests to current sample
            for(int t=0; t<num_tests; ++t){
                Test cur_test = node_tests[cur_node][t];
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                int child;
                if( val < cur_test.threshold )
                    //go to the left child
                    child = 0;
                else
                    //go to the right
                    child = 1;

                (*cur_mean)[t][child][0] += train_set[i]->dof(3); //x
                (*cur_mean)[t][child][1] += train_set[i]->dof(4); //y
                (*cur_mean)[t][child][2] += train_set[i]->dof(5); //z
                (*cur_mean)[t][child][3]++;                       //count samples

            }

        }
        //get child samples in order to calculate mean values
        #pragma omp critical
        {
            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector< std::vector<float> > > *cur_mean_local = &(node_test_child_mean_local[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        (*cur_mean)[t][child][3] +=
                            (*cur_mean_local)[t][child][3];
            }
        }

        //continue after maen is calculated
        #pragma omp barrier

        #pragma omp critical
        {
            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector< std::vector<float> > > *cur_mean_local = &(node_test_child_mean_local[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        for(int coord=0; coord<3; ++coord) // x y z only
                            if((*cur_mean)[t][child][3] > 0)
                                (*cur_mean)[t][child][coord] +=
                                    (*cur_mean_local)[t][child][coord] / (*cur_mean)[t][child][3];
            }

        }

    }


    //get std values for all possible split-childs

    //[node][test][0-1(left-right)] = std(x) + std(y) + std(z) per child node per test per node
    boost::unordered_map<TreeNode*, std::vector< std::vector<float> > > node_test_child_std;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_child_std[level_nodes[i]].resize(num_tests);
        for(int t=0; t<num_tests; ++t){
            node_test_child_std[level_nodes[i]][t].resize(2);
            for(int child=0; child<2; ++child){
                node_test_child_std[level_nodes[i]][t][child] = 0;
            }
        }
    }


#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector< std::vector<float> > > node_test_child_std_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_child_std_local[level_nodes[i]].resize(num_tests);
            for(int t=0; t<num_tests; ++t){
                node_test_child_std_local[level_nodes[i]][t].resize(2);
                for(int child=0; child<2; ++child){
                    node_test_child_std_local[level_nodes[i]][t][child] = 0;
                }
            }
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode *cur_node = train_set[i]->node;
            std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[cur_node]);
            std::vector< std::vector<float> > *cur_std_local = &(node_test_child_std_local[cur_node]);
            //apply all tests to current sample
            for(int t=0; t<num_tests; ++t){
                Test cur_test = node_tests[cur_node][t];
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                int child;
                if( val < cur_test.threshold )
                    //go to left
                    child = 0;
                else
                    //go to right
                    child = 1;

                (*cur_std_local)[t][child] +=
                    pow(train_set[i]->dof(3) - (*cur_mean)[t][child][0], 2) +  //x
                    pow(train_set[i]->dof(4) - (*cur_mean)[t][child][1], 2) +  //y
                    pow(train_set[i]->dof(5) - (*cur_mean)[t][child][2], 2);   //z

            }

        }
        #pragma omp critical
        {

            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector<float> > *cur_std_local = &(node_test_child_std_local[level_nodes[n]]);
                std::vector< std::vector<float> > *cur_std = &(node_test_child_std[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        if((*cur_mean)[t][child][3] > 0)
                            (*cur_std)[t][child] +=
                                (*cur_std_local)[t][child] / (*cur_mean)[t][child][3];
            }

        }

    }

    //compute best test using std

#pragma omp parallel
    {
    int num_leaves = 0;
    #pragma omp for schedule(dynamic)
    for(int n=0; n<level_nodes.size(); ++n){
        TreeNode *cur_node = level_nodes[n];
        std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[cur_node]);
        std::vector< std::vector<float> > *cur_std = &(node_test_child_std[cur_node]);
        float min_std = FLT_MAX;
        Test best_test;
        bool found = false;
        for(int t=0; t<num_tests; ++t){
            float total_samples_left  = (*cur_mean)[t][0][3];
            float total_samples_right = (*cur_mean)[t][1][3];
            if(total_samples_left > 0.1f && total_samples_right > 0.1f){
                float std = (*cur_std)[t][0]*total_samples_left + (*cur_std)[t][1]*total_samples_right;
                if(std < min_std){
                    min_std = std;
                    best_test = node_tests[cur_node][t];
                    found = true;
                }
            }

        }
        if(found){
            cur_node->test = best_test;
            cur_node->leaf = false;
            cur_node->left = new TreeNode;
            cur_node->right = new TreeNode;
        }
        else{
            cur_node->leaf = true;            
        }
    }

    }

//    std::cout << "best tests regression: " << omp_get_wtime() - start << std::endl;

}



// ------Regression for yaw pitch roll------ //
//if no split found a node is marked as leaf.
//best tests are stored to the node of each training sample
void HFTrain::find_regression_pose_split(TrainSet &train_set,
                                            int train_samples,
                                            const std::vector<TreeNode*> &level_nodes,
                                            boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                                            boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                                            boost::unordered_map<TreeNode*, int> &node_samples
                                            )
{

//    double start = omp_get_wtime();

    //get the mean values for all possible split-childs

    // [node][test][0-1(left-right)][0-2(x,y,z)] =
    // cos(yaw), sin(yaw), cos(pitch), sin(pitch), cos(roll), sin(roll), count samples
    int num_tests = tests_per_node_* thresholds_per_test_;
    boost::unordered_map<TreeNode*, std::vector< std::vector< std::vector<float> > > > node_test_child_mean;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_child_mean[level_nodes[i]].resize(num_tests);
        for(int t=0; t<num_tests; ++t){
            node_test_child_mean[level_nodes[i]][t].resize(2);
            for(int child=0; child<2; ++child){
                node_test_child_mean[level_nodes[i]][t][child] = std::vector<float>(7, 0);
            }
        }
    }

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector< std::vector< std::vector<float> > > > node_test_child_mean_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_child_mean_local[level_nodes[i]].resize(num_tests);
            for(int t=0; t<num_tests; ++t){
                node_test_child_mean_local[level_nodes[i]][t].resize(2);
                for(int child=0; child<2; ++child){
                    node_test_child_mean_local[level_nodes[i]][t][child] = std::vector<float>(7, 0);
                }
            }
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode *cur_node = train_set[i]->node;
            //reduce calls to hash table so to reduce execution time
            std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean_local[cur_node]);
            //apply all tests to current sample
            for(int t=0; t<num_tests; ++t){
                Test cur_test = node_tests[cur_node][t];
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                int child;
                if( val < cur_test.threshold )
                    //go to the left child
                    child = 0;
                else
                    //go to the right
                    child = 1;

                (*cur_mean)[t][child][0] += cos(train_set[i]->dof(0)); //cos(yaw)
                (*cur_mean)[t][child][1] += sin(train_set[i]->dof(0)); //sin(yaw)
                (*cur_mean)[t][child][2] += cos(train_set[i]->dof(1)); //cos(pitch)
                (*cur_mean)[t][child][3] += sin(train_set[i]->dof(1)); //sin(pitch)
                (*cur_mean)[t][child][4] += cos(train_set[i]->dof(2)); //cos(roll)
                (*cur_mean)[t][child][5] += sin(train_set[i]->dof(2)); //sin(roll)
                (*cur_mean)[t][child][6]++;                            //count samples

            }

        }
        //get child samples in order to calculate mean values
        #pragma omp critical
        {
            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector< std::vector<float> > > *cur_mean_local = &(node_test_child_mean_local[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        (*cur_mean)[t][child][6] +=
                            (*cur_mean_local)[t][child][6];
            }
        }

        //continue after mean is calculated
        #pragma omp barrier

        #pragma omp critical
        {
            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector< std::vector<float> > > *cur_mean_local = &(node_test_child_mean_local[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        for(int pose=0; pose<6; ++pose)
                            if((*cur_mean)[t][child][6] > 0)
                                (*cur_mean)[t][child][pose] +=
                                    (*cur_mean_local)[t][child][pose] / (*cur_mean)[t][child][6];
            }

        }

    }


    //get std values for all possible split-childs

    //[node][test][0-1(left-right)] = std(x) + std(y) + std(z) per child node per test per node
    boost::unordered_map<TreeNode*, std::vector< std::vector<float> > > node_test_child_std;
    for(int i=0; i<level_nodes.size(); ++i){
        node_test_child_std[level_nodes[i]].resize(num_tests);
        for(int t=0; t<num_tests; ++t){
            node_test_child_std[level_nodes[i]][t].resize(2);
            for(int child=0; child<2; ++child){
                node_test_child_std[level_nodes[i]][t][child] = 0;
            }
        }
    }


#pragma omp parallel
    {

        boost::unordered_map<TreeNode*, std::vector< std::vector<float> > > node_test_child_std_local;
        for(int i=0; i<level_nodes.size(); ++i){
            node_test_child_std_local[level_nodes[i]].resize(num_tests);
            for(int t=0; t<num_tests; ++t){
                node_test_child_std_local[level_nodes[i]][t].resize(2);
                for(int child=0; child<2; ++child){
                    node_test_child_std_local[level_nodes[i]][t][child] = 0;
                }
            }
        }

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            TreeNode *cur_node = train_set[i]->node;
            std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[cur_node]);
            std::vector< std::vector<float> > *cur_std_local = &(node_test_child_std_local[cur_node]);
            //apply all tests to current sample
            for(int t=0; t<num_tests; ++t){
                Test cur_test = node_tests[cur_node][t];
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                int child;
                if( val < cur_test.threshold )
                    //go to left
                    child = 0;
                else
                    //go to right
                    child = 1;

                (*cur_std_local)[t][child] +=
                        pow(cos(train_set[i]->dof(0)) - (*cur_mean)[t][child][0], 2) +  //cos(yaw)
                        pow(sin(train_set[i]->dof(0)) - (*cur_mean)[t][child][1], 2) +  //sin(yaw)
                        pow(cos(train_set[i]->dof(1)) - (*cur_mean)[t][child][2], 2) +  //cos(pitch)
                        pow(sin(train_set[i]->dof(1)) - (*cur_mean)[t][child][3], 2) +  //sin(pitch)
                        pow(cos(train_set[i]->dof(2)) - (*cur_mean)[t][child][4], 2) +  //cos(roll)
                        pow(sin(train_set[i]->dof(2)) - (*cur_mean)[t][child][5], 2);   //sin(roll)

            }

        }
        #pragma omp critical
        {

            for(int n=0; n<level_nodes.size(); ++n){
                std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[level_nodes[n]]);
                std::vector< std::vector<float> > *cur_std_local = &(node_test_child_std_local[level_nodes[n]]);
                std::vector< std::vector<float> > *cur_std = &(node_test_child_std[level_nodes[n]]);
                for(int t=0; t<num_tests; ++t)
                    for(int child=0; child<2; ++child)
                        if((*cur_mean)[t][child][6] > 0)
                            (*cur_std)[t][child] +=
                                (*cur_std_local)[t][child] / (*cur_mean)[t][child][6];
            }

        }

    }

    //compute best test using std

#pragma omp parallel
    {
    int num_leaves = 0;
    #pragma omp for schedule(dynamic)
    for(int n=0; n<level_nodes.size(); ++n){
        TreeNode *cur_node = level_nodes[n];
        std::vector< std::vector< std::vector<float> > > *cur_mean = &(node_test_child_mean[cur_node]);
        std::vector< std::vector<float> > *cur_std = &(node_test_child_std[cur_node]);
        float min_std = FLT_MAX;
        Test best_test;
        bool found = false;
        for(int t=0; t<num_tests; ++t){
            float total_samples_left  = (*cur_mean)[t][0][6];
            float total_samples_right = (*cur_mean)[t][1][6];
            if(total_samples_left > 0.1f && total_samples_right > 0.1f){
                float std = (*cur_std)[t][0]*total_samples_left + (*cur_std)[t][1]*total_samples_right;
                if(std < min_std){
                    min_std = std;
                    best_test = node_tests[cur_node][t];
                    found = true;
                }
            }

        }
        if(found){
            cur_node->test = best_test;
            cur_node->leaf = false;
            cur_node->left = new TreeNode;
            cur_node->right = new TreeNode;
        }
        else{
            cur_node->leaf = true;            
        }
    }

    }

//    std::cout << "best tests regression: " << omp_get_wtime() - start << std::endl;

}





void HFTrain::apply_tests_to_train_samples(TrainSet &train_set, int train_samples, boost::unordered_map<TreeNode*, int> &new_node_samples)
{
    //apply splitting with best test found and count new node samples

    omp_set_num_threads(threads_per_tree_);

#pragma omp parallel
    {
        boost::unordered_map<TreeNode*, int> new_node_samples_local;

        #pragma omp for schedule(dynamic)
        for(int i=0; i<train_samples; ++i){
            if(!train_set[i]->node->leaf){
                //set node number for new level
                Test cur_test = train_set[i]->node->test;
                float val;
                if(cur_test.measure_mode == 0)
                    val = train_set[i]->feature_vector(cur_test.feature1) - train_set[i]->feature_vector(cur_test.feature2);
                else if(cur_test.measure_mode == 1)
                    val = train_set[i]->feature_vector(cur_test.feature1);

                if(val < cur_test.threshold)
                    train_set[i]->node = train_set[i]->node->left;
                else
                    train_set[i]->node = train_set[i]->node->right;

                if(new_node_samples_local.count(train_set[i]->node))
                    new_node_samples_local[train_set[i]->node]++;
                else
                    new_node_samples_local[train_set[i]->node] = 1;

            }
        }

        #pragma omp critical
        {

            for(boost::unordered_map<TreeNode*, int>::iterator t = new_node_samples_local.begin(); t != new_node_samples_local.end(); ++t){
                if(new_node_samples.count(t->first))
                    new_node_samples[t->first] += t->second;
                else
                    new_node_samples[t->first] = t->second;
            }

        }

    }

//    std::cout << "New node samples: " << new_node_samples[train_set[0]->node] << " " << new_node_samples[level_nodes[0]->right] << std::endl;

}



void HFTrain::optimize_level(TrainSet &train_set,
                             int train_samples,                             
                             int cur_level,
                             const std::vector<TreeNode*> &level_nodes,
                             boost::unordered_map<TreeNode*, int> &new_node_samples
                             )
{


    std::cout << "Depth: " << cur_level << "  nodes: " << level_nodes.size() << std::endl;
//    double start = omp_get_wtime();

    //get random test features, not thresholds yet
    boost::unordered_map<TreeNode*, std::vector<Test> > node_tests;
    get_random_features(level_nodes, node_tests);


    //get min-max for each feature for each node to constraint thresholds' range
    boost::unordered_map<TreeNode*, std::vector<float> > node_test_min;
    boost::unordered_map<TreeNode*, std::vector<float> > node_test_max;
    boost::unordered_map<TreeNode*, std::vector<int> > node_class_samples;
    boost::unordered_map<TreeNode*, int> node_samples;

    get_min_max_count_samples(train_set,
                              train_samples,
                              level_nodes,
                              node_tests,
                              node_test_min,
                              node_test_max,
                              node_class_samples,
                              node_samples);

    get_random_thresholds(level_nodes, node_test_min, node_test_max, node_tests);


    //0->classification
    //1->regression of x,y,z
    //2->refression of yaw,pitch,roll
    int split_method;
    if(cur_level < 4)
        split_method = 0;
    else
        split_method = rand() % 3;

//    std::cout << "Before split: " << omp_get_wtime() - start << std::endl;

    if(split_method == 0){
        //do classification
        find_classification_split(train_set,
                                  train_samples,
                                  level_nodes,
                                  node_tests,
                                  node_class_samples,
                                  node_samples);
    } else if(split_method == 1){
        //do regression of object coordinates
        find_regression_location_split(train_set,
                                  train_samples,
                                  level_nodes,
                                  node_tests,
                                  node_class_samples,
                                  node_samples);
    } else if(split_method == 2){
        //do regression yaw pitch roll
        find_regression_pose_split(train_set,
                                  train_samples,
                                  level_nodes,
                                  node_tests,
                                  node_class_samples,
                                  node_samples);
    }

    //apply best tests that were stored to the nodes of each train sample
    apply_tests_to_train_samples(train_set, train_samples, new_node_samples);

//    std::cout << "After split " << split_method << ": " << omp_get_wtime() - start << std::endl;

}



//train depth by depth, all nodes together!
void HFTrain::train_tree(TreeNode *root, TrainSet& train_set){

    //every training sample is set to the root
    for(int i=0; i<train_set.size(); ++i)
        train_set[i]->node = root;

    //get 66.6% of training samples at beginning
    int train_samples = 2.0f/3.0f*(float)train_set.size();
    std::vector<int> samples_per_class;
    suffle_training_set(train_set, train_samples, samples_per_class);

    //add the root
    std::vector<TreeNode*> cur_level_nodes;
    cur_level_nodes.push_back(root);

    int cur_level = 0;
    int cur_leaf_id = 0;
    while(cur_level_nodes.size() > 0) {

        boost::unordered_map<TreeNode*, int> new_node_samples;
        optimize_level(train_set, train_samples, cur_level, cur_level_nodes, new_node_samples);

        std::vector<TreeNode*> new_level_nodes;
        for(int i=0; i<cur_level_nodes.size(); ++i){
            if(!cur_level_nodes[i]->leaf){
                if(new_node_samples[cur_level_nodes[i]->left] > min_samples_)
                    new_level_nodes.push_back(cur_level_nodes[i]->left);
                else{
                    //make leaf left
                    num_leaves_++;
                    cur_level_nodes[i]->left->leaf = true;                    
                }
                if(new_node_samples[cur_level_nodes[i]->right] > min_samples_)
                    new_level_nodes.push_back(cur_level_nodes[i]->right);
                else{
                    //make leaf right
                    num_leaves_++;
                    cur_level_nodes[i]->right->leaf = true;                    
                }

            }
        }
//        double start = omp_get_wtime();
        make_leafs(train_set, train_samples, samples_per_class, cur_leaf_id);
//        std::cout << "Leafs made in: " << omp_get_wtime() - start << std::endl;

        //put train samples that belong to leaf nodes to the end and reduce the size
        //of training samples accordingly
        clean_trainset(train_set, train_samples);
//        std::cout << "Train samples after clean: :" << train_samples << std::endl;

        cur_level_nodes = new_level_nodes;
        cur_level++;
//        std::cout << "Current level: " << cur_level << std::endl;
//        std::cout << "Level nodes: " << cur_level_nodes.size() << std::endl;
//        std::cout << "Total Leaf nodes: " << num_leaves_ << std::endl;

    }


}



void HFTrain::train()
{

    omp_set_nested(1);
    omp_set_num_threads(threads_for_parallel_trees_);
    #pragma omp parallel
    {
        //each thread should have different seed
        //otherwise random numbers would be the same in all tress
        srand( int(time(NULL)) ^ omp_get_thread_num() );

        //each thread should read its own input to avoid confict
        TrainSet train_set;
        int feature_vector_length, number_of_classes;
        std::stringstream msg;
        msg << "Thread " << omp_get_thread_num() << ": Reading input..." << std::endl;
        std::cout << msg.str();
        getTrainSet(train_set, number_of_classes, feature_vector_length);
        msg.str("");
        msg << "Thread " << omp_get_thread_num() << ": Finished reading " << train_set.size() << " train samples." << std::endl;
        std::cout << msg.str();

        if(omp_get_thread_num() == 0){
            number_of_classes_ = number_of_classes;
            feature_vector_length_ = feature_vector_length;
            //write forest.txt
            std::ofstream finfo((output_folder_ + "/forest.txt").c_str());
            finfo << number_of_trees_ << " " <<
                     number_of_classes_ << " " <<
                     feature_vector_length_ << " " <<
                     patch_size_in_voxels_ << " " <<
                     voxel_size_in_m_ << std::endl;
            finfo.close();
        }
        //wait until member variables are written
        #pragma omp barrier

        //start training
        #pragma omp for schedule(dynamic)
        for(int t=start_tree_no_; t< start_tree_no_ + number_of_trees_; ++t){

            std::stringstream s;
            s << output_folder_ << "/tree" << t << ".dat";
            std::ofstream f(s.str().c_str(), std::ios::out | std::ios::binary);
            CHECK(f) << "Could not write to file " << s.str();

            TreeNode *tree = new TreeNode;

            msg.str("");
            msg << "Training tree: " << t << std::endl;
            std::cout << msg.str();
            double t0 = omp_get_wtime();
            train_tree(tree, train_set);
            double t1 = omp_get_wtime();

            saveTreeNode(tree, f);
            msg.str("");
            msg << "Tree " << t << " saved - " << t1-t0 << "sec" << std::endl;
            std::cout << msg.str();

            deleteTreeNode(tree);
            delete tree;
        }

    } //omp parallel
	
}


