/*****************************
Ben Lee (benlee@ece.ucsb.edu)

Instance Set

*****************************/
#include "librf/instance_set.h"
#include <fstream>
#include <float.h>
#include "librf/weights.h"
#include "librf/types.h"

namespace librf {

const int InstanceSet::kLeft = 0;
const int InstanceSet::kRight = 1;
InstanceSet::InstanceSet(){}


InstanceSet::InstanceSet(const string& filename, int num_features) : attributes_(num_features) {
    // default to libsvm reading now
    ifstream in(filename.c_str());
    string line;
    while(getline(in,line)) {
       Instance i(line, num_features);
       labels_.push_back(i.true_label);
       distribution_.add(i.true_label);
       // Add features into attributes list
       for (int j = 0; j < num_features; ++j) {
          attributes_[j].push_back(i.features_[j]);
       }
    }
    cout << "InstanceSet Distribution:" << endl;
    distribution_.print();
   cout << "Sorting indices " << endl;
    create_sorted_indices();
    create_rank_array();
    cout << "Instance Set loaded. " << endl;
}

void InstanceSet::create_bootstrap(vector<float> * bs) {
    // randomly sample WITH replacement for the length of the set
    // returns a float vector of WEIGHTS (default weight is zero);
    // Allocate and zero weight vector
    for (int i = 0; i < instances_.size(); ++i) {
      bs->push_back(0.0);
    }
    // Random sampling
    for (int i = 0; i < instances_.size(); ++i) {
        int selected = rand() % instances_.size();
       (*bs)[selected] += 1.0;
    }
}

void InstanceSet::create_sorted_indices() {
    // allocate sorted_indices_
    sorted_indices_.resize(attributes_.size());
    // sort 
    for (int i = 0; i < attributes_.size(); ++i) {
        sort_attribute(attributes_[i], &sorted_indices_[i]);
    }
}

void InstanceSet::create_rank_array() {
  //allocate properly sized ranks
  ranks_ = new uint16*[attributes_.size()];
  for (int i = 0; i < attributes_.size(); ++i) {
    ranks_[i] = new uint16[attributes_[i].size()];
  }
  // for each attribute
  // get the sorted index order
  for (int i = 0; i < attributes_.size(); ++i) {
    const vector<int>& sorted_order = sorted_indices_[i];
    float last_value = attributes_[i][sorted_order[0]];
    uint16 rank = 0;
    ranks_[i][sorted_order[0]] = rank;

    for (int j = 1; j < sorted_order.size(); ++j) {
      float current_value = attributes_[i][sorted_order[j]];
      if (current_value > last_value) {
        rank++;
        last_value = current_value;
      }
      ranks_[i][sorted_order[j]] = rank;
  }
  }
}
void InstanceSet::sort_attribute(const vector<float>& attribute,
                                 vector<int>*indices) {
    vector<pair<float, int> > pairs;
    for (int i = 0; i < attribute.size(); ++i) {
        pairs.push_back(make_pair(attribute[i],i));
    }
    sort(pairs.begin(), pairs.end());
    for (int i = 0; i < pairs.size(); ++i) {
        indices->push_back(pairs[i].second);
    }
}
// Using the weights as a mask on the instance set -
// find the best split with respect to this attribute
void InstanceSet::find_best_split_for_attr(int attr,
		// const vector<float>& weights,
    const weight_list& weights,
    float prior_entropy,
		float *split_point, float *split_gain) const {
		// Class distribution of the split 
		DiscreteDist split_dist[2];
    const int kLeft = 0;
    const int kRight = 1;
		// Move all the instances into RIGHT split at first
		for (int i = 0; i < weights.size(); ++i) {
			split_dist[kRight].add(labels_[i],weights[i]);
		}
		float best_gain = - DBL_MAX;
		// float prior_entropy = split_dist[0].entropy_over_classes();
    // Use the sorted indices
    const vector<int>& indices = get_sorted_indices(attr);
		int split_index = 0;

    // More intuitive, but naive version
    // Get all the splits possible -- mid-points between DISTINCT values
    vector <float> possible_splits;
    float last_value = attributes_[attr][indices[0]];
    // cout << "possible splits on " << attr <<endl;
    for (int i = 1; i <indices.size(); ++i) {
      int j = indices[i];
      float value = attributes_[attr][j];
      if (value != last_value) {
        float s = (value + last_value)/2.0;
        // cout << s << endl;
        possible_splits.push_back(s);
        last_value = value;
      }
    }
    //  Out of possible splits, find the best one
    //  MOVE EVERYTHING THAT IS <= THAN SPLIT POINT INTO LEFT SET
    int l = 0;
    for (int k = 0; k < possible_splits.size(); ++k) {
      float split = possible_splits[k];
      for (; l < indices.size() &&
             attributes_[attr][indices[l]] < split; ++l) {
        int inst = indices[l];
        split_dist[kRight].remove(labels_[inst], weights[inst]);
				split_dist[kLeft].add(labels_[inst], weights[inst]);
      }
      float split_entropy = DiscreteDist::entropy_conditioned(split_dist, 2);
      float curr_gain = prior_entropy - split_entropy;
      if (curr_gain > best_gain) {
          best_gain = curr_gain;
          *split_point = split;
      }
    }
		*split_gain = best_gain;
}


// Find the best split from list of attributes
void InstanceSet::find_best_split(const vector<int>& attrs,
                //                  const vector<float>& weights,
                                 const weight_list& weights,
																 float prior_entropy,
								int* split_attr, float *split_point, float* split_gain) const{
	float best_gain = -DBL_MAX;
	int best_attr = -1;
	float best_split = -DBL_MAX;

	for (int i = 0; i < attrs.size(); ++i) {
    int attr =attrs[i];
		float curr_split;
		float curr_gain;
		find_best_split_for_attr(attr, weights, prior_entropy,
                             &curr_split, &curr_gain);
    // cout << attr << ":" << curr_split << "->" << curr_gain <<endl;
		if (curr_gain > best_gain) {
				best_gain = curr_gain;
				best_split = curr_split;
				best_attr = attr;
		}
	}
	*split_attr = best_attr;
	*split_point = best_split;
	*split_gain = best_gain;
}



void InstanceSet::split_data(const weight_list& parent,
                        int split_attr, float split_point,
                        weight_list*& left, weight_list*& right) const{
   // Calculate new distributions
   DiscreteDist split_dist[2];
   for (int i =0; i < parent.size(); ++i) {
     byte weight = parent[i];
     if (weight > 0) {
       float value = attributes_[split_attr][i];
       if (value < split_point) {
          split_dist[kLeft].add(labels_[i], weight);
       } else {
          split_dist[kRight].add(labels_[i], weight);
       }
      }
   }
   // Allocate weight lists
   left = new weight_list(size(), split_dist[kLeft].sum());
   right = new weight_list(size(), split_dist[kRight].sum());
   for (int i =0; i < parent.size(); ++i) {
     byte weight = parent[i];
     if (weight > 0) {
       float value = attributes_[split_attr][i];
       if (value < split_point) {
          left->add(i, weight);
       } else {
          right->add(i, weight);
       }
     }
   }
   // cout << "left:" << left->sum() <<endl;
   // cout << "right:" << right->sum() <<endl;
}


// Grab a subset of the instance (for getting OOB data
// Seems like maybe this should be a static named constructor
InstanceSet::InstanceSet(const InstanceSet& set,
                         const weight_list& weights) : attributes_(set.num_attributes()){
  // Calculate the number of OOB cases
  //cout << "creating OOB subset for weight list of size "
  //     << weights.size() << endl;
  for (int i = 0; i < weights.size(); ++i) {
    if (weights[i] == 0) {
      // append instance
      for (int j = 0; j < set.num_attributes(); ++j) {
          attributes_[j].push_back(set.get_attribute(i, j));
      }
      labels_.push_back(set.label(i));
    }
  }
}

void InstanceSet::permute(int var, unsigned int *seed) {
  vector<float>& attr = attributes_[var];
  for (int i = 0; i < attr.size(); ++i) {
    int idx = rand_r(seed) % labels_.size(); // randomly select an index
    float tmp = attr[i];  // swap last value with random index value
    attr[i] = attr[idx];
    attr[idx] =  tmp;
  }
}

void InstanceSet::load_var(int var, const vector<float>& source) {
  // use the STL built-in copy/assignment
  attributes_[var] = source;
}

void InstanceSet::save_var(int var, vector<float>* target) {
  const vector<float>& attr = attributes_[var];
  // use the STL built-in copy/assignment
  *target = attr;
}

} // namespace
