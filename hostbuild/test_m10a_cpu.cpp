#include "../app/src/main/cpp/tensor/tensor.h"
#include "../app/src/main/cpp/operators/cpu_operators.h"
#include "../app/src/main/cpp/diffusion/diffusion_runtime.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace localimage::tensor;
static bool closef(float a,float b,float eps=1e-4f){return std::fabs(a-b)<=eps*(1.0f+std::fabs(b));}
static int fail(const char* m){std::cerr<<"FAIL: "<<m<<"\n";return 1;}
int main(){
 std::string e; TensorRuntime rt;
 auto a=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e); auto b=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e);
 if(!a.valid()||!b.valid()) return fail("tensor alloc");
 float av[]={1,2,3,4}, bv[]={5,6,7,8}; std::memcpy(a.mutableData(),av,sizeof(av)); std::memcpy(b.mutableData(),bv,sizeof(bv));
 Tensor o;
 if(!localimage::ops::add(a,b,o,e)||!closef(((float*)o.data())[3],12)) return fail("add");
 if(!localimage::ops::sub(a,b,o,e)||!closef(((float*)o.data())[0],-4)) return fail("sub");
 if(!localimage::ops::mul(a,b,o,e)||!closef(((float*)o.data())[3],32)) return fail("mul");
 if(!localimage::ops::div(b,a,o,e)||!closef(((float*)o.data())[2],7.0f/3.0f)) return fail("div");
 Tensor m1=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e),m2=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e);
 std::memcpy(m1.mutableData(),av,sizeof(av));std::memcpy(m2.mutableData(),bv,sizeof(bv));
 if(!localimage::ops::matmul(m1,m2,o,e)) return fail("matmul");
 if(!closef(((float*)o.data())[0],19)||!closef(((float*)o.data())[1],22)||!closef(((float*)o.data())[2],43)||!closef(((float*)o.data())[3],50)) return fail("matmul values");
 auto g=rt.createTensor(TensorShape({2}),TensorDType::F32,e); ((float*)g.mutableData())[0]=1;((float*)g.mutableData())[1]=1;
 auto x=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e);float xv[]={1,3,5,7};std::memcpy(x.mutableData(),xv,sizeof(xv));
 if(!localimage::ops::layerNorm(x,g,nullptr,1e-5,o,e)) return fail("layernorm");
 if(!closef(((float*)o.data())[0],-1,2e-3f)||!closef(((float*)o.data())[1],1,2e-3f)) return fail("layernorm values");
 auto gn=rt.createTensor(TensorShape({2}),TensorDType::F32,e);((float*)gn.mutableData())[0]=1;((float*)gn.mutableData())[1]=1;
 auto gx=rt.createTensor(TensorShape({1,2,1,1}),TensorDType::F32,e);float gv[]={1,3};std::memcpy(gx.mutableData(),gv,sizeof(gv));
 if(!localimage::ops::groupNorm(gx,gn,nullptr,1,1e-5,o,e)) return fail("groupnorm");
 auto smx=rt.createTensor(TensorShape({1,3}),TensorDType::F32,e);float sv[]={1,2,3};std::memcpy(smx.mutableData(),sv,sizeof(sv));
 if(!localimage::ops::softmax(smx,1,o,e)) return fail("softmax");double ss=0;for(int i=0;i<3;i++)ss+=((float*)o.data())[i];if(!closef((float)ss,1))return fail("softmax sum");
 if(!localimage::ops::transpose(smx,{1,0},o,e)||o.shape().dims()!=std::vector<uint64_t>({3,1}))return fail("transpose");
 auto c1=rt.createTensor(TensorShape({1,2}),TensorDType::F32,e),c2=rt.createTensor(TensorShape({1,1}),TensorDType::F32,e);((float*)c1.mutableData())[0]=1;((float*)c1.mutableData())[1]=2;((float*)c2.mutableData())[0]=3;
 if(!localimage::ops::concat({c1,c2},1,o,e)||o.shape().dim(1)!=3||!closef(((float*)o.data())[2],3))return fail("concat");
 auto up=rt.createTensor(TensorShape({1,1,1,2}),TensorDType::F32,e);((float*)up.mutableData())[0]=4;((float*)up.mutableData())[1]=5;if(!localimage::ops::upsampleNearest(up,2,o,e)||o.shape().dims()!=std::vector<uint64_t>({1,1,2,4}))return fail("upsample");
 auto q=rt.createTensor(TensorShape({1,2,2}),TensorDType::F32,e),k=rt.createTensor(TensorShape({1,2,2}),TensorDType::F32,e),v=rt.createTensor(TensorShape({1,2,2}),TensorDType::F32,e);float qv[]={1,0,0,1};float kv[]={1,0,0,1};float vv[]={1,2,3,4};std::memcpy(q.mutableData(),qv,sizeof(qv));std::memcpy(k.mutableData(),kv,sizeof(kv));std::memcpy(v.mutableData(),vv,sizeof(vv));if(!localimage::ops::scaledDotProductAttention(q,k,v,1.0/std::sqrt(2.0),nullptr,o,e))return fail("attention");for(uint64_t i=0;i<o.shape().elementCount();++i)if(!std::isfinite(((float*)o.data())[i]))return fail("attention finite");
 auto f16=rt.createTensor(TensorShape({2}),TensorDType::F16,e);if(!f16.valid()||!f16.writeFloat32(0,1.5f,e)||!f16.writeFloat32(1,-2.25f,e)||!closef(f16.readFloat32(0,e),1.5f,2e-3f))return fail("f16");
 auto bf=rt.createTensor(TensorShape({2}),TensorDType::BF16,e);if(!bf.valid()||!bf.writeFloat32(0,1.5f,e)||!closef(bf.readFloat32(0,e),1.5f,2e-2f))return fail("bf16");
 std::cout<<"M10-A CPU numerical suite: PASS\n";
 std::cout<<"F32/F16/BF16 tensor storage: PASS\n";
 std::cout<<"MatMul expected [[19,22],[43,50]]: PASS\n";
 std::cout<<"LayerNorm/GroupNorm/Softmax/Attention: PASS\n";
 return 0;
}
