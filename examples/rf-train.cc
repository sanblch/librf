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
    SwitchArg csvFlag("","csv","Data is a CSV file",false);
    SwitchArg headerFlag("","header","CSV file has a var name header",false);
    ValueArg<string> delimArg("","delim","CSV delimiter", false,",","delimiter");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "trainingdata");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "rfmodel");
    ValueArg<string> labelArg("l", "label",
                              "Label file", false, "", "labels");
    ValueArg<int> numfeaturesArg("f", "features", "# features", false,
                                 -1, "int");
    ValueArg<int> treesArg("t", "trees", "# Trees", false, 10, "int");
    ValueArg<int> kArg("k", "vars", "# vars per tree", false,
                                 10, "int");
    cmd.add(delimArg);
    cmd.add(headerFlag);
    cmd.add(csvFlag);
    cmd.add(labelArg);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.add(treesArg);
    cmd.add(kArg);

    cmd.parse(argc, argv);
    bool csv = csvFlag.getValue();
    bool header = headerFlag.getValue();
    string delim = delimArg.getValue();
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    string labelfile = labelArg.getValue();
    int K = kArg.getValue();
    int num_features = numfeaturesArg.getValue();
    int num_trees = treesArg.getValue();
    InstanceSet* set = NULL;
    if (!csv) {
      set = InstanceSet::load_libsvm(datafile, num_features);
    } else {
      set = InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);
    }
    RandomForest rf(*set, num_trees, K, 1000);
    cout << "Training Accuracy " << rf.training_accuracy() << endl;
    cout << "OOB Accuracy " << rf.oob_accuracy() << endl;
    ofstream out(modelfile.c_str());
    rf.write(out);
    cout << "Model file saved to " << modelfile << endl;
    // rf.print();
    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
