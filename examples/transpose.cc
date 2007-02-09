#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "librf/librf.h"
#include "librf/instance_set.h"
using namespace std;
using namespace librf;

int main(int argc, char* argv[]) {
  // open instance set
  string base(argv[1]);
  InstanceSet* set =
      InstanceSet::load_csv_and_labels(string(argv[1]),
                                       string(argv[2]), false, " ");
  // write the transposed
  string transposed = base + ".transposed";
  ofstream out(transposed.c_str());
  set->write_transposed_csv(out, string(" "));
  return 0;
}
