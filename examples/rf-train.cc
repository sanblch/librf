#include "librf/librf.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <math.h>
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
                                 -1, "int");
    ValueArg<string> probArg("p", "prob",
                              "probability file", false, "", "probs");
    cmd.add(delimArg);
    cmd.add(headerFlag);
    cmd.add(csvFlag);
    cmd.add(labelArg);
    cmd.add(numfeaturesArg);
    cmd.add(dataArg);
    cmd.add(modelArg);
    cmd.add(treesArg);
    cmd.add(kArg);
    cmd.add(probArg);
    cmd.parse(argc, argv);

    bool csv = csvFlag.getValue();
    bool header = headerFlag.getValue();
    string delim = delimArg.getValue();
    string datafile = dataArg.getValue();
    string modelfile = modelArg.getValue();
    string labelfile = labelArg.getValue();
    string probfile = probArg.getValue();
    int K = kArg.getValue();
    int num_features = numfeaturesArg.getValue();
    int num_trees = treesArg.getValue();
    InstanceSet* set = NULL;
    if (!csv) {
      set = InstanceSet::load_libsvm(datafile, num_features);
    } else {
      set = InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);
    }
    // if mtry was not set defaults to sqrt(num_features)
    if (K == -1) {
       K = int(sqrt(double(set->num_attributes())));
    }
    // vector<int> weights;
    RandomForest rf(*set, num_trees, K); //, weights);
    cout << "Training Accuracy " << rf.training_accuracy() << endl;
    cout << "OOB Accuracy " << rf.oob_accuracy() << endl;
    cout << "---Confusion Matrix----" << endl;
    rf.oob_confusion();
    vector<pair<float, float> > rd;
    vector<int> hist;
    cout << "Reliability Diagram" << endl;
    rf.reliability_diagram(10, &rd, &hist);
    for (int i = 0; i < rd.size(); ++i) {
      cout << rd[i].first << " " << rd[i].second << " " << hist[i] << endl;
    }

    ofstream out(modelfile.c_str());
    rf.write(out);
    cout << "Model file saved to " << modelfile << endl;

    if (probfile.size() > 0) {
      ofstream prob_out(probfile.c_str());
      for (int i = 0; i < set->size(); i++) {
        prob_out << rf.oob_predict_prob(i, 1) << endl;
      }
    }
    // rf.print();
    delete set;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
