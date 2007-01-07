/*****************************
Ben Lee (benlee@ece.ucsb.edu)

Instance Set


*****************************/
#ifndef _INSTANCE_SET_H_
#define _INSTANCE_SET_H_

#include <string>
#include <vector>
#include "librf/instance.h"
#include "librf/discrete_dist.h"

using namespace std;

namespace librf {

class weight_list;

class InstanceSet {
    public:
        InstanceSet(const InstanceSet&, const weight_list&);
        InstanceSet();
        InstanceSet(const string &filename, int num);
        // copy a variable array out 
        void save_var(int var, vector<float> *target);
        void load_var(int var, const vector<float>&);
        // permute a variable's instances (shuffle)
        void permute(int var, unsigned int * seed);
        void create_bootstrap(vector<float> * bs);
        void create_sorted_indices();
        const vector<int>& get_sorted_indices(int attribute) const{
            return sorted_indices_[attribute];
        }
        int mode_label() const {
          return distribution_.mode();
        }
        unsigned char label(int i) const{
          return labels_[i];
        }
        unsigned int size() const {
            return labels_.size();
        }
        unsigned int num_attributes() const {
          return attributes_.size();
        }
        float get_attribute(int i, int attr) const {
          return attributes_[attr][i];
        }
        uint16 get_rank(int i, int attr) const {
          return ranks_[attr][i];
        }
				// single attribute find best split
				void find_best_split_for_attr(int attr,
                                      const weight_list& weights,
                                      // const vector<float>& weights,
																		  float prior_entropy,
																			float *best_split, float*gain) const;
				// find best split with given attributes
				void find_best_split(const vector<int>&attrs,
                             const weight_list& weights,
                             // const vector<float>& weights,
                             float prior_entropy,
                             int* split_attr, float *best_split, float* gain) const;
        float class_entropy() const{
          return distribution_.entropy_over_classes();
        }
        void split_data(const weight_list& parent,
                        int split_attr, float split_point,
                        weight_list*& left, weight_list*& right) const;
    private:
        const static int kLeft;
        const static int kRight;
        void create_rank_array();
        void sort_attribute(const vector<float>&attribute, vector<int>*indices);
        DiscreteDist distribution_;
        vector<Instance> instances_;
        // List of Attribute Lists
        // Thus access is attributes_ [attribute] [ instance]
        vector< vector<float> > attributes_;
        uint16 **ranks_;
//        vector< vector<unsigned int> > ranks_;
        // List of true labels
        // access is labels_ [instance]
        vector<unsigned char> labels_;
        vector< vector<int> > sorted_indices_;
};

}  // namespace
#endif
