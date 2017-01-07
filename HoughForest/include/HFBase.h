/* Copyright (C) 2016 Andreas Doumanoglou 
 * You may use, distribute and modify this code under the terms
 * included in the LICENSE.txt
 *
 * Base class for Hough Tree
 */

#ifndef HFBASE_H_
#define HFBASE_H_

#include <vector>
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <boost/unordered_map.hpp>

class HFBase
{

protected:

    struct Test{

        //0-> 2 pixel test, 1-> 1 pixel test
        int measure_mode;
        int feature1, feature2;
        float threshold;

        Test& operator=(const Test& t){
            this->measure_mode = t.measure_mode;
            this->feature1 = t.feature1;
            this->feature2 = t.feature2;
            this->threshold = t.threshold;
        }

    };

    struct TreeNode{

        bool leaf;
        int leaf_id;
        Test test;
        TreeNode *left, *right;        
        std::vector<float> class_prob;
        //6 dimensional vector, yaw-pitch-roll-x-y-z
        //list for each class, 2 modes are usually stored
        std::vector< std::vector<Eigen::VectorXf> > hough_votes;
        //mean and std of each feature dimension
        std::vector<std::vector<float> > mean_feature_vector;
        std::vector<std::vector<float> > std_feature_vector;

        TreeNode(){
            leaf = false;
            left = 0;
            right = 0;
            leaf_id = -1;
        }

    };

    struct HoughPoint {

        std::vector<int> d;

        HoughPoint(int _d0, int _d1, int _d2, int _d3, int _d4, int _d5) {

            d.resize(6);
            d[0] = _d0;
            d[1] = _d1;
            d[2] = _d2;
            d[3] = _d3;
            d[4] = _d4;
            d[5] = _d5;

        }

        HoughPoint(){
            d.resize(6, 0);
        }

        //needed to be used as key to map.
        //a map will be used as a sparse matrix to store hough votes
        //and find the modes
        bool operator<(const HoughPoint& hp) const
        {
           for(int i=0; i<5; ++i)
               if(d[i] != hp.d[i])
                   return (d[i] < hp.d[i]);

           return (d[5] < hp.d[5]);
        }

        bool operator==(const HoughPoint& hp) const{

            bool eq = true;
            for(int i=0; i<6; ++i)
                eq &= d[i] == hp.d[i];
            return eq;
        }

        bool operator=(const HoughPoint& hp){
            d = hp.d;
        }

    };


    struct HpHash{
        std::size_t operator()(const HoughPoint& h) const {
            return boost::hash_range(h.d.begin(), h.d.end());
        }
    };

    typedef boost::unordered_map<HoughPoint, float, HpHash> Hough3DMap;


    int number_of_classes_;
    int number_of_trees_;
    int feature_vector_length_;
    float voxel_size_in_m_;
    int patch_size_in_voxels_;
    std::vector<TreeNode*> trees_;
    bool is_forest_loaded_;

    void saveTreeNode(TreeNode *n, std::ofstream& f);
    void deleteTreeNode(TreeNode *n);
    bool loadNodeFromFile(TreeNode *n, std::ifstream& f);
    bool loadForestFromFolder(const std::string &folder_name);

public:
    bool isForestLoaded() const { return is_forest_loaded_; }

};

#endif
