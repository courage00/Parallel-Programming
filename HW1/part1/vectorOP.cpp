#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_float zerof = _pp_vset_float(0);
  __pp_vec_float oneF = _pp_vset_float(1.f);
  __pp_vec_float nineF = _pp_vset_float(9.999999f);
  __pp_vec_int oneInt = _pp_vset_int(1);


  __pp_mask maskAll, maskIsNegative, maskIsNotNegative,maskIsPrevntOver,maskCnt;
  int vectorNum =VECTOR_WIDTH;

  if(vectorNum>N){ //防止N小於VECTOR_WIDTH
    vectorNum = N;
  }
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones(vectorNum);// All ones

    maskIsNegative = _pp_init_ones(0);// All zeros

    _pp_vload_float(x, values + i, maskAll);// float x = values[i];
    _pp_vload_int(y, exponents + i, maskAll);// int y = exponents[i]; as count

    _pp_veq_int(maskIsNegative,y,zero,maskAll);// if (y == 0) {
    _pp_vstore_float(output + i, oneF, maskIsNegative);//   output[i] = 1.f; }
                                          //1
    maskIsPrevntOver = _pp_mask_not(maskAll);//防止多餘執行次數
    maskIsNegative = _pp_mask_or(maskIsNegative,maskIsPrevntOver);

    maskIsNotNegative = _pp_mask_not(maskIsNegative);// else{
    maskIsNegative = _pp_init_ones(0);//初始化

    _pp_vload_float(result, values + i, maskIsNotNegative);//   float result = x;
    maskCnt = maskIsNotNegative;
    _pp_vsub_int(y, y, oneInt, maskCnt);//count--;
    while (_pp_cntbits(maskCnt) > 0) {
      _pp_vgt_int(maskCnt,y,zero,maskCnt);//y>count=0 
      _pp_vmult_float(result,result,x,maskCnt);//     result *= x;
      _pp_vsub_int(y, y, oneInt, maskCnt);//count--; 
    }
    _pp_vgt_float(maskIsNegative,result,nineF,maskIsNotNegative);//if (result > 9.999999f){

    _pp_vadd_float(result,zerof, nineF, maskIsNegative);//     result = 9.999999f;}
    _pp_vstore_float(output + i, result, maskIsNotNegative);//   output[i] = result;}
    if((2*vectorNum)+i>=N){ //防止不是VECTOR_WIDTH倍數
      vectorNum = N -i-vectorNum;
    }
  }
}


// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}
