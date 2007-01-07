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
    CmdLine cmd("rf-train", ' ', "0.1");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "string");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "string");
    ValueArg<int> numfeaturesArg("f", "features", "# features", true,
                                 -1, "int");
    ValueArg<int> treesArg("t", "trees", "# Trees", false, 10, "int");
    ValueArg<int> kArg("k", "vars", "# vars per tree", false,
                                 10, "int");
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.add(numfeaturesArg);
    cmd.add(treesArg);
    cmd.add(kArg);

    cmd.parse(argc, argv);
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    int K = kArg.getValue();
    int num_features = numfeaturesArg.getValue();
    int num_trees = treesArg.getValue();

    InstanceSet set(datafile, num_features);
    RandomForest rf(set, num_trees, K, 12);
    cout << "Training Accuracy " << rf.training_accuracy() << endl;
    cout << "OOB Accuracy " << rf.oob_accuracy() << endl;
    ofstream out(modelfile.c_str());
    rf.write(out);
    cout << "Model file saved to " << modelfile << endl;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
