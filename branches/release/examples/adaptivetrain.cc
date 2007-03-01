#include "librf/librf.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <set>
#include <vector>

using namespace std;
using namespace librf;
using namespace TCLAP;
int main(int argc, char*argv[]) {
  // Check arguments
  try {
    SwitchArg headerFlag("","header","CSV file has a var name header",false);
    ValueArg<string> delimArg("","delim","CSV delimiter", false,",","delimiter");
    ValueArg<string> labelArg("l", "label",
                              "Label file", false, "", "labels");
    CmdLine cmd("rf-predict", ' ', "0.1");
    ValueArg<string>  dataArg("d", "data",
                                   "Training Data", true, "", "testdata");
    ValueArg<string> outputArg("o", "output", "predictions", true, "", "output");
    ValueArg<int> treesArg("t", "trees", "# Trees", false, 10, "int");
    ValueArg<int> kArg("k", "vars", "# vars per tree", false, 10, "int");

    cmd.add(delimArg);
    cmd.add(labelArg);
    cmd.add(outputArg);
    cmd.add(dataArg);
    cmd.add(treesArg);
    cmd.add(kArg);

    cmd.parse(argc, argv);

    bool header = headerFlag.getValue();
    string delim = delimArg.getValue();
    string labelfile = labelArg.getValue();
    string datafile = dataArg.getValue();
    string outfile = outputArg.getValue();
    int K = kArg.getValue();
    int num_trees = treesArg.getValue();
    InstanceSet* iset = InstanceSet::load_csv_and_labels(datafile, labelfile, header, delim);
    RandomForest rf(*iset, num_trees, K);
    cout << "OOB accuracy: " << rf.oob_accuracy() << endl;
    set<int> patterns_used;
    map<int, int> clocks_used;
    map<pair<int,int>, int> pattern_clocks_used;
    int sum_pc = 0;
    for (int i = 0; i < iset->size(); ++i) {
      vector<pair<int, float> > nodes_used;
      rf.oob_predict(i, &nodes_used);
      // insert patterns
      set<pair<int,int> > pc_used;
      for (int j = 0; j < nodes_used.size(); ++j) {
        patterns_used.insert(nodes_used[j].first);
        int clock = int(floor(nodes_used[j].second/50.0));
        pair<int, int> p = make_pair(nodes_used[j].first, clock);
        if (pattern_clocks_used.find(p) == pattern_clocks_used.end()) {
          pattern_clocks_used[p] = 1;
        } else {
          pattern_clocks_used[p]++;
        }
        if (clocks_used.find(clock) == clocks_used.end()) {
          clocks_used[clock] = 1;
        } else {
          clocks_used[clock]++;
        }
        pc_used.insert(make_pair(nodes_used[j].first, clock));
      }
      sum_pc += pc_used.size();
    }
    cout << "Totally used patterns: " << patterns_used.size() << endl;
    cout << "Totally used pattern clock combos: " << pattern_clocks_used.size() << endl;
    cout << "Total used clocks: " << clocks_used.size() << endl;
    cout << "TOTAL test time: " << sum_pc << endl;
    cout << "Average test time per chip: " << float(sum_pc)/iset->size() << endl;
    // sort pattern_clock combos by use
    vector<pair<int, pair<int, int> > > sorted;
    for (map<pair<int, int>, int>::iterator it = pattern_clocks_used.begin();
         it != pattern_clocks_used.end(); ++it) {
      sorted.push_back(make_pair(it->second, it->first));
    }
    sort(sorted.begin(),sorted.end(), greater<pair<int, pair<int,int> > >());
    for (int i = 0; i < sorted.size(); ++i) {
      cout << sorted[i].second.first << " ";
      cout << sorted[i].second.second << " ";
      cout << sorted[i].first << endl;
    }
    //for (map<int,int>::iterator it = clocks_used.begin();
    //     it != clocks_used.end(); ++it) {
    //  cout << it->first << " " << it->second << endl;
    //}
    delete iset;
  }
  catch (TCLAP::ArgException &e)  // catch any exceptions 
  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
