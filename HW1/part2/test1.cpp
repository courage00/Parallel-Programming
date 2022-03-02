#include <iostream>
#include "test.h"
#include "fasttime.h"

typedef float float16 __attribute__((ext_vector_type(16)));
void test1(float* __restrict a, float* __restrict b, float* __restrict c, int N){
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16); 
  c = (float *)__builtin_assume_aligned(c, 16); 
  float16* f = (float16*)a ;
  float16* h = (float16*)b ;
  float16* k = (float16*)c ;

  fasttime_t time1 = gettime();
  for (int i=0; i<I; i++) {
    for (int j=0; j<N/16; j++) {
      k[j] = f[j] + h[j];
    }
  }
  fasttime_t time2 = gettime();

  double elapsedf = tdiff(time1, time2);
  std::cout << "Elapsed execution time of the loop in test1():\n" 
    << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}
