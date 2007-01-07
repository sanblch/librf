#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include <vector>

using namespace std;

namespace librf {

class Instance;
class InstanceSet;
class Tree;

class RandomForest {
  public:
    RandomForest();
    RandomForest(const InstanceSet& set,
                 int num_trees,
                 int K,
                 int max_depth);
    ~RandomForest();
        // predict a new instance 
     int predict(const Instance& c) const;
     float predict_prob(const Instance& c) const;
     // predict an instance from a set        
     int predict(const InstanceSet& set, int instance_no) const;
     float testing_accuracy(const InstanceSet& testset) const;
     float training_accuracy() const;
     float oob_accuracy() const;
     void variable_importance(vector< pair<float, int> >* ranking,
                              unsigned int* seed) const;
     void read(istream& i);
     void write(ostream& o);
     void print() const;
  private:
    const InstanceSet& set_;
    vector<Tree*> trees_;
    int max_depth_;
    int K_;
};

} // namespace
#endif
