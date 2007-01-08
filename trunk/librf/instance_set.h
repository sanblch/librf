/**
 * @file
 * @brief Instance Set
 * This is the abstraction for a data set
 * -- Currently libSVM
 * -- want to support CSV, ARFF
 */
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
        InstanceSet();
        static InstanceSet* create_subset(const InstanceSet&, const weight_list&);
        static InstanceSet* load_from_csv_and_labels(const string& data,
                                                    const string& labels,
                                                    bool header,
                                                    const string& delim);
        static InstanceSet* load_from_libsvm(const string& data,
                                            int num_features);
        // copy a variable array out 
        void save_var(int var, vector<float> *target);
        void load_var(int var, const vector<float>&);
        // permute a variable's instances (shuffle)
        void permute(int var, unsigned int * seed);
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
        //float class_entropy() const{
        //  return distribution_.entropy_over_classes();
        //}
    private:
        /// Load from csv file and labels
        InstanceSet(const string& csv_data, const string& labels,
                    bool header=false, const string& delim=",");
        /// Load from libsvm format
        InstanceSet(const string& filename, int num);
        /// Get a subset of an existing instance set
        InstanceSet(const InstanceSet&, const weight_list&);
        void load_labels(istream& in);
        void load_csv(istream& in, bool header, const string& delim);
        void load_svm(istream& in);
        void create_dummy_var_names(int n);
        void sort_attribute(const vector<float>&attribute, vector<int>*indices);
        DiscreteDist distribution_;
        vector<Instance> instances_;
        // List of Attribute Lists
        // Thus access is attributes_ [attribute] [ instance]
        vector< vector<float> > attributes_;
        // List of true labels
        // access is labels_ [instance]
        vector<unsigned char> labels_;
        vector<string> var_names_;
        vector< vector<int> > sorted_indices_;
};

}  // namespace
#endif
