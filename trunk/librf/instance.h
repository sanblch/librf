/*****************************
Ben Lee (benlee@ece.ucsb.edu)

A dense row... naive implementation
******************************/
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

namespace librf {

template <class out_type, class in_value>
out_type cast_stream(const in_value & t) {
  stringstream ss;
  ss << t; // first insert value to stream
  out_type result; // value will be converted to out_type
  ss >> result; // write value to result
  return result;
}



struct Instance {
Instance(const string& line, int num_features) : features_(num_features,0.0){
  stringstream ss(line);
  string pair_str;
  float label;
  ss >> label;
  if (label == -1.0) {
      true_label =0;
  } else if (label ==0.0) {
      true_label =0;
  } else if (label ==1.0) {
      true_label =1;
  } else {
      cerr << "Incorrect label (only +1,0,-1 supported)" << endl;
      assert(false);
  }
  while (ss >> pair_str) {
    string::size_type pos = pair_str.find_first_of(':');
    string first = pair_str.substr(0,pos);
    string second = pair_str.substr(pos+1);
    pair<int,float> p = make_pair(cast_stream<int>(first), cast_stream<float>(second));
    if (p.first < num_features) {
      features_[p.first] = p.second;
    } else {
       cout << line << endl;
       assert(false);
    }
  }
}
     // Dense versus Sparse (the eternal question) 
    // map<int, float> features_; //feature values
    vector<float> features_; // feature values
    bool true_label;  // true label
};

} //namespace
