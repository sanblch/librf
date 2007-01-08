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
    CmdLine cmd("rf-predict", ' ', "0.1");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "testdata");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "rfmodel");

    ValueArg<int> numfeaturesArg("f", "features", "# features", true,
                                 -1, "int");
    ValueArg<string> outputArg("o", "output", "predictions", true, "", "output");
    cmd.add(outputArg);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.parse(argc, argv);
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    string outfile = outputArg.getValue();

    int num_features = numfeaturesArg.getValue();

    InstanceSet* set = InstanceSet::load_libsvm(datafile, num_features);
    RandomForest rf;
    ifstream in(modelfile.c_str());
    rf.read(in);
    cout << "Test accuracy: " << rf.testing_accuracy(*set) << endl;;
    ofstream out(outfile.c_str());
    for (int i = 0; i < set->size(); ++i) {
      out << rf.predict_prob(*set, i, 1) << endl;
    }
    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
