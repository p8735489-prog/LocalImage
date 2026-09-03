#include "../app/src/main/cpp/tensor/tensor.h"
#include "../app/src/main/cpp/operators/cpu_operators.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace localimage::tensor;
int main(){
 std::string e; TensorRuntime rt;
 Tensor x=rt.createTensor(TensorShape({2,3}),TensorDType::F32,e); assert(x.valid());
 for(uint64_t i=0;i<6;i++) assert(x.writeFloat32(i,(float)i,e));
 Tensor t; assert(rt.transpose(x,{1,0},t,e)); assert(t.shape().dims()==std::vector<uint64_t>({3,2})); assert(std::fabs(t.readFloat32(1,e)-3.f)<1e-6);
 Tensor s; assert(rt.slice(x,1,1,2,s,e)); assert(s.shape().dims()==std::vector<uint64_t>({2,2})); assert(std::fabs(s.readFloat32(0,e)-1.f)<1e-6);
 Tensor one=rt.createTensor(TensorShape({2,1}),TensorDType::F32,e); assert(one.valid());
for(uint64_t i=0;i<2;i++) assert(one.writeFloat32(i,(float)i,e));
Tensor b; assert(localimage::ops::broadcastTo(one,TensorShape({2,3}),b,e)); assert(b.shape().dims()==std::vector<uint64_t>({2,3}));
 Tensor n=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e); assert(n.valid());
 for(uint64_t i=0;i<4;i++) assert(n.writeFloat32(i,(float)i,e));
 Tensor u; assert(localimage::ops::upsampleNearest(n,2,u,e)); assert(u.shape().dims()==std::vector<uint64_t>({1,1,4,4}));
 std::cout<<"M10-B transform CPU: PASS\n"; return 0;
}
