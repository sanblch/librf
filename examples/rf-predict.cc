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
                                   "Training Data", true, "", "string");
    ValueArg<string> modelArg("m", "model",
                              "Model file output", true, "", "string");

    ValueArg<int> numfeaturesArg("f", "features", "# features", true,
                                 -1, "int");
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.add(numfeaturesArg);
    cmd.parse(argc, argv);
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    int num_features = numfeaturesArg.getValue();

    InstanceSet set(datafile, num_features);
    RandomForest rf;
    ifstream in(modelfile.c_str());
    rf.read(in);
    rf.testing_accuracy(set);
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
