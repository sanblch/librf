#include "librf/random_forest.h"
#include "librf/instance_set.h"
#include <UnitTest++.h>
#include <iostream>
#include <fstream>
using namespace std;
using namespace librf;


struct RF_TrainPredictFixture {
  RF_TrainPredictFixture() {
    cout << "loading heart data" << endl;
    heart_ = new InstanceSet("../data/heart.svm", 14);
  }
  ~RF_TrainPredictFixture() {
    delete heart_;
  }
  InstanceSet* heart_;
};

TEST_FIXTURE(RF_TrainPredictFixture, TrainPredictCheck) {

  RandomForest rf(*heart_, 10, 12, 12);
  rf.print();
 cout << "Training accuracy " << rf.training_accuracy() <<endl;
  cout << "OOB Accuracy " << rf.oob_accuracy() <<endl;
  cout << "Test accuracy " << rf.testing_accuracy(*test_) <<endl;
  ofstream out("out.test");
  cout <<"save test" << endl;
  rf.write(out);
  ifstream in ("out.test");
  RandomForest loaded;
  loaded.read(in);
  unsigned int seed = 1;
  vector< pair<float, int> > scores;
  rf.variable_importance(&scores, &seed);
  for (int i = 0; i < scores.size(); ++i) {
    cout << scores[i].second << ":" << scores[i].first <<endl;
  }
}
/*
int main()
{
    return UnitTest::RunAllTests();
}*/

