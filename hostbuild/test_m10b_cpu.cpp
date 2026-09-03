#include <cmath>
#include <iostream>
#include <string>
#include "../app/src/main/cpp/tensor/tensor.h"
#include "../app/src/main/cpp/operators/cpu_operators.h"
static bool close(float a,float b,float eps=1e-5f){return std::fabs(a-b)<=eps*(1+std::fabs(b));}
int main(){using namespace localimage::tensor;using namespace localimage::ops;TensorRuntime rt;std::string e;
 Tensor a=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e),b=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e);float av[]={1,2,3,4},bv[]={5,6,7,8};std::copy(av,av+4,(float*)a.mutableData());std::copy(bv,bv+4,(float*)b.mutableData());Tensor o;
 if(!matmul(a,b,o,e)){std::cerr<<e;return 1;}float ex[]={19,22,43,50};for(int i=0;i<4;i++)if(!close(((float*)o.data())[i],ex[i]))return 2;
 Tensor x=rt.createTensor(TensorShape({1,2,2,2}),TensorDType::F32,e),g=rt.createTensor(TensorShape({2}),TensorDType::F32,e);float xv[]={1,2,3,4,5,6,7,8};std::copy(xv,xv+8,(float*)x.mutableData());std::fill((float*)g.mutableData(),(float*)g.mutableData()+2,1);if(!groupNorm(x,g,nullptr,1,1e-5,o,e))return std::cerr<<e,3;
 Tensor sm=rt.createTensor(TensorShape({1,2,3}),TensorDType::F32,e);float sv[]={1,2,3,100,101,102};std::copy(sv,sv+6,(float*)sm.mutableData());if(!softmax(sm,2,o,e))return std::cerr<<e,4;for(int r=0;r<2;r++){double s=0;for(int j=0;j<3;j++)s+=((float*)o.data())[r*3+j];if(!close((float)s,1))return 5;}
 Tensor q=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e),k=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e),v=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e);float qv[]={1,0,0,1};std::copy(qv,qv+4,(float*)q.mutableData());std::copy(qv,qv+4,(float*)k.mutableData());float vv[]={1,2,3,4};std::copy(vv,vv+4,(float*)v.mutableData());if(!scaledDotProductAttention(q,k,v,1/std::sqrt(2.0),nullptr,o,e))return std::cerr<<e,6;if(o.shape().dims()!=std::vector<uint64_t>({1,1,2,2}))return 7;
 Tensor xr=rt.createTensor(TensorShape({1,2,2}),TensorDType::F32,e);float xrv[]={1,0,0,1};std::copy(xrv,xrv+4,(float*)xr.mutableData());Tensor angles=rt.createTensor(TensorShape({2,1}),TensorDType::F32,e);float ang[]={0,1.57079632679f};std::copy(ang,ang+2,(float*)angles.mutableData());Tensor ropeOut;if(!rope(xr,angles,ropeOut,e))return std::cerr<<e,8;
 Tensor f16=rt.createTensor(TensorShape({4}),TensorDType::F16,e),f32=rt.createTensor(TensorShape({4}),TensorDType::F32,e);for(int i=0;i<4;i++)f32.mutableData();for(int i=0;i<4;i++)if(!f16.writeFloat32(i,float(i)+.25f,e))return 9;Tensor f16back;if(!rt.convertDtype(f16,TensorDType::F32,f16back,e))return std::cerr<<e,10;for(int i=0;i<4;i++)if(!close(f16back.readFloat32(i,e),float(i)+.25f,2e-3f))return 11;
 Tensor bf16=rt.createTensor(TensorShape({2}),TensorDType::BF16,e);if(!bf16.valid()||!bf16.writeFloat32(0,1.25f,e)||!bf16.writeFloat32(1,-2.5f,e))return 12;if(std::fabs(bf16.readFloat32(0,e)-1.25f)>0.01f||std::fabs(bf16.readFloat32(1,e)+2.5f)>0.01f)return 13;

 // Conv2D: 1x1 input with a 2x2 kernel, no padding/stride 1.
 Tensor cx=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e), cw=rt.createTensor(TensorShape({1,1,2,2}),TensorDType::F32,e), cb=rt.createTensor(TensorShape({1}),TensorDType::F32,e), co;
 float cxv[]={1,2,3,4}, cwv[]={1,0,0,1}; std::copy(cxv,cxv+4,(float*)cx.mutableData()); std::copy(cwv,cwv+4,(float*)cw.mutableData()); ((float*)cb.mutableData())[0]=0.5f;
 if(!conv2d(cx,cw,&cb,1,0,1,1,co,e)) return std::cerr<<e,14;
 if(co.shape().dims()!=std::vector<uint64_t>({1,1,1,1}) || std::fabs(((float*)co.data())[0]-5.5f)>1e-5f) return 15;
 // LayerNorm known two-element row.
 Tensor lx=rt.createTensor(TensorShape({1,2}),TensorDType::F32,e), lg=rt.createTensor(TensorShape({2}),TensorDType::F32,e), lo;
 ((float*)lx.mutableData())[0]=1; ((float*)lx.mutableData())[1]=3; std::fill((float*)lg.mutableData(),(float*)lg.mutableData()+2,1.0f);
 if(!layerNorm(lx,lg,nullptr,1e-5,lo,e)) return std::cerr<<e,16;
 if(std::fabs(((float*)lo.data())[0]+0.999995f)>1e-3f || std::fabs(((float*)lo.data())[1]-0.999995f)>1e-3f) return 17;
 // RoPE must use position axis, not batch axis. Two batches share the same positions.
 Tensor r2=rt.createTensor(TensorShape({2,2,2}),TensorDType::F32,e), a2=rt.createTensor(TensorShape({2,1}),TensorDType::F32,e), ro2;
 float r2v[]={1,0,0,1, 2,0,0,2}, a2v[]={0,1.57079632679f}; std::copy(r2v,r2v+8,(float*)r2.mutableData()); std::copy(a2v,a2v+2,(float*)a2.mutableData());
 if(!rope(r2,a2,ro2,e)) return std::cerr<<e,18;
 if(std::fabs(((float*)ro2.data())[0]-1)>1e-5f || std::fabs(((float*)ro2.data())[3]-0)>1e-3f || std::fabs(((float*)ro2.data())[4]-2)>1e-5f) return 19;
 std::cout<<"M10-B CPU reference: PASS\nMatMul: PASS\nConv2D: PASS\nLayerNorm: PASS\nGroupNorm: PASS\nSoftmax numerical stability: PASS\nRank4 Attention: PASS\nRoPE position semantics: PASS\nF16 storage conversion: PASS\nBF16 storage conversion: PASS\n";return 0;}
