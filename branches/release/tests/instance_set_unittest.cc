#include "librf/instance_set.h"
#include <UnitTest++.h>
#include <iostream>
using namespace std;
using namespace librf;
struct InstanceSetFixture {
	// Do some Setup
	InstanceSetFixture() {
    // Load example libSVM file
    libsvm = InstanceSet::load_libsvm("../data/heart.svm", 14);
    csv = InstanceSet::load_csv_and_labels("../data/heart.csv",
                                           "../data/heart_labels.txt",true);
  }
	// Do some Teardown
	~InstanceSetFixture() {
    delete libsvm;
    delete csv;
	}
  InstanceSet* libsvm;
  InstanceSet* csv;
};

TEST_FIXTURE(InstanceSetFixture, SanityCheck)
{
  CHECK_EQUAL(libsvm->size(), 270);
  CHECK_EQUAL(csv->size(), 270);
}


/*
int main()
{
    return UnitTest::RunAllTests();
}
*/

