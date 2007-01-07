#include "instance_set.h"
#include "weights.h"
#include <UnitTest++.h>
#include <iostream>
using namespace std;

struct InstanceSetFixture {
	// Do some Setup
	InstanceSetFixture() {
    // Load example libSVM file
		is_ = new InstanceSet("data/test.svm",5);
		is2_ = new InstanceSet("data/test2.svm",5);
		is3_ = new InstanceSet("data/tree.svm",2166);
	}
	// Do some Teardown
	~InstanceSetFixture() {
		delete is_;
		delete is2_;
		delete is3_;
	}
		InstanceSet *is_;
		InstanceSet *is2_;
		InstanceSet *is3_;
};

TEST_FIXTURE(InstanceSetFixture, SanityCheck)
{
	// Make sure the file loaded properly
	 CHECK_EQUAL(is_->size(), 2);
	 vector<int> indices = is_->get_sorted_indices(4);
	 const int order[2] = {0,1};
	 CHECK_ARRAY_EQUAL(indices, order, 2);
   weight_list weights(2,2);
	 weights.add(0);
	 weights.add(1);
	 float best_split, best_gain;
   float prior_entropy = 1;
   is_->find_best_split_for_attr(4,weights, prior_entropy,
                                 &best_split,
                                 &best_gain);
	 CHECK_CLOSE(0.5, best_split, 0.01);
   int split_attr;
   cerr << "-------------test2.svm-------------" << endl;
   prior_entropy = is2_->class_entropy();
   cerr << "Prior entropy is: " << prior_entropy << endl;
   vector<int> attrs;
   attrs.push_back(0);
   attrs.push_back(1);
   attrs.push_back(2);
   attrs.push_back(3);
   attrs.push_back(4);
   weight_list weights2(4,4);
	 weights2.add(0);
	 weights2.add(1);
	 weights2.add(2);
	 weights2.add(3);
   is2_->find_best_split(attrs, weights2, prior_entropy,
                        &split_attr, &best_split, &best_gain);
   cerr << "Best split using attribute: " << split_attr <<endl;
   cerr << "Split point: " << best_split <<endl;
/*
  weight_list weights3(8,2);
  weights3.add(6);
  weights3.add(7);
  vector<int> attrs2;
  attrs2.push_back(0);
  attrs2.push_back(1);
  attrs2.push_back(2);
  is3_->find_best_split(attrs2, weights3, 1, &split_attr,
                        &best_split, &best_gain);
  cerr << "Best split using attribute: " << split_attr <<endl;
  cerr << "Split point: " << best_split <<endl;
  */
}


/*
int main()
{
    return UnitTest::RunAllTests();
}
*/

