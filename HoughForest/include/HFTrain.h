/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * HFTrain class for training a hough forest in parallel.
 * Parallelization is happening across all the samples of all
 * the nodes of each depth of the tree, so that we gain as much
 * as possible from the parallelization that is independent from
 * the number of nodes per depth, or the number of samples per
 * node.
 */
#ifndef HFTRAIN_H_
#define HFTRAIN_H_

#include <time.h>
#include <HFBase.h>
#include <boost/unordered_map.hpp>


class HFTrain : public HFBase
{   

protected:

    struct TrainSample{

        int class_no;
        //used during training
        TreeNode *node;
        //store 6DoF information
        //0->yaw, 1->pitch, 2->roll
        //3->x, 4->y, 5->z
        Eigen::VectorXf dof;
        Eigen::VectorXf feature_vector;

    };

    typedef std::vector<TrainSample*> TrainSet;

    std::string input_file_;
    std::string output_folder_;    
    int min_samples_;
    int tests_per_node_;    
    int thresholds_per_test_;
    int threads_per_tree_;
    int threads_for_parallel_trees_;
    int num_leaves_;
    int start_tree_no_;    
    int patch_size_in_voxels_;
    float voxel_size_in_m_;

    void getTrainSet(TrainSet &train_set, int &number_of_classes, int &feature_vector_length);


public:

    HFTrain()
        : min_samples_(30)
        , tests_per_node_(10)
        , thresholds_per_test_(10)
        , input_file_("")
        , num_leaves_(0)        
        , threads_for_parallel_trees_(1)
        , threads_per_tree_(1)
        , start_tree_no_(0)
        , patch_size_in_voxels_(8) // default values, should be set correctly
        , voxel_size_in_m_(0.005)  // in order the detection to work. They are only
                                   // used to write them to forest.txt
    {
        number_of_trees_ = 3;
    }

    void train();

    void setNumTrees(int ntrees){
        number_of_trees_ = ntrees;
    }

    void setMinSamples(int min_samples){
        min_samples_ = min_samples;
    }

    void setTestsPerNode(int tests_per_node){
        tests_per_node_ = tests_per_node;
    }

    void setThresPerTest(int thresholds_per_test){
        thresholds_per_test_ = thresholds_per_test;
    }

    void setInputPatchesFilename(const std::string &filename){
        input_file_ = filename;
    }

    void setOutputFolder(const std::string &out_folder){
        output_folder_ = out_folder;
    }

    void setNumberOfThreads(int threads_per_tree, int threads_for_parallel_trees){
        threads_per_tree_ = threads_per_tree;
        threads_for_parallel_trees_ = threads_for_parallel_trees;
    }

    void setStartTreeNo(int start_tree_no){
        start_tree_no_ = start_tree_no;
    }

    void setPatchSizeInVoxels(int ps) {
        patch_size_in_voxels_ = ps;
    }

    void setVoxelSizeInM(float vs) {
        voxel_size_in_m_ = vs;
    }


private:    

    void growNode(TreeNode *root, TrainSet& train_set);
    void train_tree(TreeNode *root, TrainSet& train_set);
    void optimize_level(TrainSet &train_set, int train_samples, int cur_level, const std::vector<TreeNode *> &level_nodes, boost::unordered_map<TreeNode *, int> &new_node_samples);
    void clean_trainset(TrainSet& train_set, int &train_samples);        
    void make_leafs(TrainSet &train_set, int train_samples, const std::vector<int> &samples_per_class, int &leaf_id);
    void suffle_training_set(TrainSet &train_set, int train_samples, std::vector<int> &samples_per_class);

    void get_min_max_count_samples(const TrainSet &train_set,
                              int train_samples,
                              const std::vector<TreeNode*> &level_nodes,
                              boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                              boost::unordered_map<TreeNode*, std::vector<float> > &node_test_min,
                              boost::unordered_map<TreeNode*, std::vector<float> > &node_test_max,
                              boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                              boost::unordered_map<TreeNode*, int> &node_samples
                              );

    void get_random_features(const std::vector<TreeNode*> &level_nodes,
                             boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests
                             );

    void get_random_thresholds(const std::vector<TreeNode*> &level_nodes,
                               boost::unordered_map<TreeNode*, std::vector<float> > &node_test_min,
                               boost::unordered_map<TreeNode*, std::vector<float> > &node_test_max,
                               boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests
                               );

    void find_classification_split(TrainSet &train_set,
                                    int train_samples,
                                    const std::vector<TreeNode*> &level_nodes,
                                    boost::unordered_map<TreeNode *,
                                    std::vector<Test> > &node_tests,
                                    boost::unordered_map<TreeNode *,
                                    std::vector<int> > &node_class_samples,
                                    boost::unordered_map<TreeNode *, int> &node_samples);

    void apply_tests_to_train_samples(TrainSet &train_set, int train_samples, boost::unordered_map<TreeNode*, int> &new_node_samples);

    void find_regression_location_split(TrainSet &train_set,
                                                int train_samples,
                                                const std::vector<TreeNode*> &level_nodes,
                                                boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                                                boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                                                boost::unordered_map<TreeNode*, int> &node_samples
                                                );

    void find_regression_pose_split(TrainSet &train_set,
                                    int train_samples,
                                    const std::vector<TreeNode*> &level_nodes,
                                    boost::unordered_map<TreeNode*, std::vector<Test> > &node_tests,
                                    boost::unordered_map<TreeNode*, std::vector<int> > &node_class_samples,
                                    boost::unordered_map<TreeNode*, int> &node_samples
                                    );


};

#endif
