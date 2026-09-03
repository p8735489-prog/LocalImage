#include <cassert>
#include <cmath>
#include <iostream>
#include "../app/src/main/cpp/operators/cpu_operators.h"
#include "../app/src/main/cpp/tensor/tensor.h"
using namespace localimage::tensor;
int main(){
 std::string e;
 TensorRuntime rt;
 Tensor x=rt.createTensor(TensorShape({5}),TensorDType::F32,e), y;
 assert(x.valid());
 float v[5]={-3,-1,0,1,3}; for(int i=0;i<5;i++) ((float*)x.mutableData())[i]=v[i];
 assert(localimage::ops::unary(x,y,"gelu",e));
 for(int i=0;i<5;i++){float ex=0.5f*v[i]*(1.0f+std::erf(v[i]/std::sqrt(2.0f))); assert(std::fabs(((float*)y.data())[i]-ex)<1e-6f);}
 std::cout<<"GELU CPU golden: PASS\n";
 return 0;
}
