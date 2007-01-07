
#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>
#include <stdlib.h>

namespace librf {

void random_sample(int n, int K, vector<int>*v, unsigned int* seed) {
  if (K < n) {
  int pop = n;
  v->reserve(K);
  for (int i = K; i > 0; --i) {
    float cumprob = 1.0;
    float x = float(rand_r(seed))/RAND_MAX;
    for (; x < cumprob; pop--) {
      cumprob -= cumprob * i /pop;
    }
    v->push_back(n - pop - 1);
  }
  } else {
    for (int i =0; i < n; i++) {
      v->push_back(i);
    }
  }
}

} // namespace
#endif
