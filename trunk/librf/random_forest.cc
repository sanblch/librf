#include "librf/random_forest.h"
#include "librf/tree.h"
#include "librf/instance_set.h"
#include "librf/weights.h"
#include <fstream>
#include <algorithm>

namespace librf {

RandomForest::RandomForest() : set_(InstanceSet()) {}
RandomForest::RandomForest(const InstanceSet& set,
                           int num_trees,
                           int K,
                           int max_depth) :set_(set),
                                           K_(K),
                                           max_depth_(max_depth){
  // cout << "RandomForest Constructor " << num_trees << endl;
  for (int i = 0; i < num_trees; ++i) {
    weight_list* w = new weight_list(set.size(), set.size());
    // sample with replacement
    for (int j = 0; j < set.size(); ++j) {
      w->add(rand()%set.size());
    }
    Tree* tree = new Tree(set, w,  K, max_depth);
    tree->grow();
    // cout << "Grew tree " << i << endl;
    trees_.push_back(tree);
  }
}

RandomForest::~RandomForest() {
  for (int i = 0; i < trees_.size(); ++i) {
    delete trees_[i];
  }
}

void RandomForest::print() const {
  for (int i = 0; i < trees_.size(); ++i) {
     trees_[i]->print();
  }
}

void RandomForest::write(ostream& o) {
  o << trees_.size() << " " << K_ << " " << max_depth_ << endl;
  for (int i = 0; i < trees_.size(); ++i) {
    trees_[i]->write(o);
  }
}

void RandomForest::read(istream& in) {
  int num_trees, K, max_depth;
  in >> num_trees >> K >> max_depth;
  for (int i = 0; i < num_trees; ++i) {
    trees_.push_back(new Tree(in));
  }
}

int RandomForest::predict(const InstanceSet& set, int instance_no) const {
  // Gather the votes from each tree
  DiscreteDist votes;
  for (int i = 0; i < trees_.size(); ++i) {
    int predict = trees_[i]->predict(set, instance_no);
    votes.add(predict);
  }
  return votes.mode();
}

float RandomForest::oob_accuracy() const {
  weight_list correct(set_.size(), set_.size());
  weight_list incorrect(set_.size(), set_.size());

  for (int i = 0; i < trees_.size(); ++i) {
    trees_[i]->oob_cases(&correct, &incorrect);
  }
  int total = 0;
  for (int i = 0; i < set_.size(); ++i) {
    if (correct[i] > incorrect[i]) {
      total++;
    }
  }
  return float(total)/set_.size();
}

/*
int RandomForest::predict(const Instance& c) const {
  // Gather the votes from each tree
  DiscreteDist votes;
  for (int i = 0; i < trees_.size(); ++i) {
    int predict = trees_[i]->predict(c);
    votes.add(predict);
  }
  return votes.mode();
}*/

float RandomForest::training_accuracy() const {
  int correct = 0;
  for (int i =0; i < set_.size(); ++i) {
    if (predict(set_, i) == set_.label(i))
      correct++;
  }
  return float(correct) / set_.size();
}

float RandomForest::testing_accuracy(const InstanceSet& set) const {
  int correct = 0;
  for (int i = 0; i < set.size(); ++i) {
    if (predict(set, i) == set.label(i))
      correct++;
  }
  return float(correct) / set.size();
}



void RandomForest::variable_importance(vector< pair< float, int> >*ranking,
                                       unsigned int* seed) const {
  vector< pair<float, int> > importances; // sum, count
  // Zero-out importances
  for (int i = 0; i < set_.num_attributes(); ++i) {
    importances.push_back(make_pair(float(0.0),int(0)));
  }
  for (int i = 0; i < trees_.size(); ++i) {
    map<int, float> tree_importance;
    // gather scores from individual tree
    trees_[i]->variable_importance(&tree_importance, seed);
    //aggregate
    for (map<int,float>::const_iterator i = tree_importance.begin();
         i != tree_importance.end(); ++i) {
      importances[i->first].first += i->second;
      importances[i->first].second ++;
    }
  }
  // Get the mean of scores
  vector<float> raw_scores;
  float sum = 0;
  float sum_of_squares = 0;
  for (int i = 0; i < importances.size(); ++i) {
    float avg = 0;
    if (importances[i].second != 0) {
      avg = importances[i].first / importances[i].second;
    }
    assert(avg == avg);
    raw_scores.push_back(avg);
    sum += avg;
    sum_of_squares += (avg * avg);
  }
  float mean = sum / importances.size();
  assert(mean == mean);
  float std = sqrt(sum_of_squares/importances.size() - mean*mean);
  assert(std == std);

  // Write the z-scores
  for (int i = 0; i < raw_scores.size(); ++i) {
    float raw = raw_scores[i];
    float zscore = 0;
    if (std != 0) {
      zscore = (raw - mean) / std;
    }
    assert(zscore == zscore);
    ranking->push_back(make_pair(zscore, i));
  }
  // Sort
  sort(ranking->begin(), ranking->end(), greater<pair<float,int> >());
}

/*
float RandomForest::predict_prob(const Instance& c) const {
  // Gather the votes from each tree
  DiscreteDist votes;
  for (int i = 0; i < trees_.size(); ++i) {
    int predict = trees_[i]->predict(c);
    votes.add(predict);
  }
  int count = votes.weight(1);
  return float(count) / trees_.size();
}*/

} // namespace
