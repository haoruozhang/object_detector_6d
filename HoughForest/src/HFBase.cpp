#include <sstream>
#include <HFBase.h>

void HFBase::saveTreeNode(TreeNode *n, std::ofstream& f){

	f.write((char*)&n->leaf, sizeof(bool));			
	
	if(n->leaf){
		
        f.write((char*)&n->leaf_id, sizeof(int));

        for(int i=0; i<number_of_classes_; ++i)
            f.write((char*)&n->class_prob[i], sizeof(float));        


        for(int c=0; c<number_of_classes_; ++c){
            int nmodes = n->hough_votes[c].size();
            f.write((char*)&nmodes, sizeof(int));
            for(int i=0; i<nmodes; ++i)
                for(int j=0; j<6; ++j)
                    f.write((char*)&(n->hough_votes[c][i](j)), sizeof(float) );
//            for(int ft=0; ft<feature_vector_length_; ++ft){
//                f.write((char*)&n->mean_feature_vector[c][ft], sizeof(float));
//                f.write((char*)&n->std_feature_vector[c][ft], sizeof(float));
//            }

        }	
		
	}else{
        f.write((char*)&n->test.measure_mode,   sizeof(int));
        f.write((char*)&n->test.feature1,       sizeof(int));
        f.write((char*)&n->test.feature2,       sizeof(int));
        f.write((char*)&n->test.threshold,      sizeof(float));
		
		saveTreeNode(n->left, f);
		saveTreeNode(n->right, f);
	}
}

void HFBase::deleteTreeNode(TreeNode *n){

    if(n->left){
        if(!n->left->leaf)
            deleteTreeNode(n->left);

        delete(n->left);
    }

    if(n->right){
        if(!n->right->leaf)
            deleteTreeNode(n->right);

        delete(n->right);
    }

}

bool HFBase::loadNodeFromFile(TreeNode *n, std::ifstream& f){

	try{
		f.read((char*)&n->leaf, sizeof(bool));
	
		if(n->leaf){
		
            f.read((char*)&n->leaf_id, sizeof(int));

            n->class_prob.resize(number_of_classes_);
            for(int i=0; i<number_of_classes_; ++i)
                f.read((char*)&n->class_prob[i], sizeof(float));

            n->hough_votes.resize(number_of_classes_);
//            n->mean_feature_vector.resize(number_of_classes_);
//            n->std_feature_vector.resize(number_of_classes_);
            for(int c=0; c<number_of_classes_; ++c){
                int nmodes;
                f.read((char*)&nmodes, sizeof(int));
                n->hough_votes[c].resize(nmodes);
                for(int i=0; i<nmodes; ++i){
                    n->hough_votes[c][i] = Eigen::VectorXf(6);
                    for(int j=0; j<6; ++j)
                        f.read((char*)&(n->hough_votes[c][i](j)), sizeof(float) );
                }
//                n->mean_feature_vector[c].resize(feature_vector_length_);
//                n->std_feature_vector[c].resize(feature_vector_length_);
//                for(int ft=0; ft<feature_vector_length_; ++ft){
//                    f.read((char*)&n->mean_feature_vector[c][ft], sizeof(float));
//                    f.read((char*)&n->std_feature_vector[c][ft], sizeof(float));
//                }
            }

			return true;
		
		}else{		
			Test test;
            f.read((char*)&test.measure_mode,   sizeof(int));
            f.read((char*)&test.feature1,       sizeof(int));
            f.read((char*)&test.feature2,       sizeof(int));
            f.read((char*)&test.threshold,      sizeof(float));
										
			n->test = test;	
            n->left = new TreeNode;
            n->right = new TreeNode;
			return loadNodeFromFile(n->left, f) & loadNodeFromFile(n->right, f);
		}
	}catch(std::exception e){
		return false;
	}
}

bool HFBase::loadForestFromFolder(const std::string &folder_name){
		
    is_forest_loaded_ = false;
	
    std::ifstream conf((folder_name+"/forest.txt").c_str());
	if(!conf.is_open())
		return false;

    conf >> number_of_trees_ >> number_of_classes_ >> feature_vector_length_ >>
            patch_size_in_voxels_ >> voxel_size_in_m_;
	conf.close();
	
    trees_.resize(number_of_trees_);
	
    for(int t=0; t<number_of_trees_; ++t){
		std::stringstream s;
        s << folder_name.c_str() << "/tree" << t << ".dat";
		std::ifstream f(s.str().c_str(), std::ios::in | std::ios::binary);

		if(f.is_open()){														

            trees_[t] = new TreeNode();
            if(!loadNodeFromFile(trees_[t], f))
				return false;

		}
		else
			return false;

		f.close();				
	}	
	
    is_forest_loaded_ = true;
	return true;

}
