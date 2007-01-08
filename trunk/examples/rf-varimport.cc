#include "librf/librf.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace librf;
using namespace TCLAP;
int main(int argc, char*argv[]) {
  // Check arguments
  try {
    CmdLine cmd("rf-varimport", ' ', "0.1");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "trainingdata");
    ValueArg<string>  labelsArg("l", "labels",
                                   "Labels File", true, "", "labels");
    ValueArg<int> treesArg("t", "trees", "# Trees", false, 10, "int");
    ValueArg<int> kArg("k", "vars", "# vars per tree", false,
                                 10, "int");
    cmd.add(dataArg);
    cmd.add(labelsArg);
    cmd.add(treesArg);
    cmd.add(kArg);

    cmd.parse(argc, argv);
    string datafile = dataArg.getValue();
    string labelfile = labelsArg.getValue();
    int K = kArg.getValue();
    int num_trees = treesArg.getValue();

    InstanceSet* set = InstanceSet::load_csv_and_labels(datafile, labelfile, true);
    RandomForest rf(*set, num_trees, K, 12);
    unsigned int seed = 1;
    vector< pair<float, int> > scores;
    rf.variable_importance(&scores, &seed);
    for (int i = 0; i < scores.size(); ++i) {
      cout << set->get_varname(scores[i].second) << ":" << scores[i].first <<endl;
    }
    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
