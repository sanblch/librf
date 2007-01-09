/** 
 * @file 
 * @brief Randomforest interface
 * This is the interface to manage a random forest
 */
#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include <vector>

using namespace std;

namespace librf {

class Instance;
class InstanceSet;
class Tree;
/**
 * @brief
 * RandomForest class.  Interface for growing random forests from training
 * data or loading a random forest from disk.
 */
class RandomForest {
  public:
    /// Empty constructor
    RandomForest();
    /// Constructor. (Build from training data)
    RandomForest(const InstanceSet& set,
                 int num_trees,
                 int K,
                 int max_depth);
    ~RandomForest();
     /// Method to predict the label
     // int predict(const Instance& c) const;
     /// Method that returns the class probability 
     // float predict_prob(const Instance& c) const;
     /// Method to predict the label
     int predict(const InstanceSet& set, int instance_no) const;
     /// Predict probability of given label
     float predict_prob(const InstanceSet& set, int instance_no, int label) const;
     /// Returns test accuracy of a labeled test set
     float testing_accuracy(const InstanceSet& testset) const;
     /// Returns training accuracy 
     float training_accuracy() const;
     /// Returns OOB accuracy (unbiased estimate of test accuracy)
     float oob_accuracy() const;
     /// Variable importance ranking of features
     void variable_importance(vector< pair<float, int> >* ranking,
                              unsigned int* seed) const;
     /// Load random forest
     void read(istream& i);
     /// Save random forest
     void write(ostream& o);
     /// Debug output
     void print() const;
  private:
    const InstanceSet& set_; // training data set
    vector<Tree*> trees_; // component trees in the forest
    int max_depth_; // maximum depth of trees
    int K_; // random vars to try per split
};

} // namespace
#endif
