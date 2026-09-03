#include "../diffusion/diffusion_runtime.h"
#include <cmath>
#include <iostream>
using namespace localimage::tensor;
using namespace localimage::diffusion;
int main(){std::string e;TensorRuntime rt;
 Tensor x=rt.createTensor(TensorShape({2,1,4}),TensorDType::F32,e),w=rt.createTensor(TensorShape({4}),TensorDType::F32,e);if(!x.valid()||!w.valid())return 1;float*X=(float*)x.mutableData();float*W=(float*)w.mutableData();for(int i=0;i<8;i++)X[i]=(float)(i-3);for(int i=0;i<4;i++)W[i]=1;Tensor o;if(!rmsNorm(x,w,1e-5,o,e)) {std::cerr<<e;return 2;}if(!std::isfinite(((float*)o.data())[0]))return 3;
 Tensor t=rt.createTensor(TensorShape({2}),TensorDType::F32,e);float*tp=(float*)t.mutableData();tp[0]=0;tp[1]=1;Tensor te;if(!timestepEmbedding(t,8,10000,te,e))return 4;if(te.shape().dim(1)!=8)return 5;
 Tensor f=rt.createTensor(TensorShape({2,2}),TensorDType::F32,e);float*fp=(float*)f.mutableData();fp[0]=0;fp[1]=1;fp[2]=0;fp[3]=1;Tensor re;if(!rotaryEmbedding(x,f,1.0,re,e))return 6;
 SchedulerConfig c;c.inference_steps=8;DDIMScheduler ddim;if(!ddim.configure(c,e))return 7;Tensor prev;if(!ddim.step(x,999,x,prev,e))return 8;EulerScheduler eu;if(!eu.configure(c,e))return 9;if(!eu.step(x,1,x,prev,e))return 10;FlowMatchScheduler fm;if(!fm.configure(c,e))return 11;if(!fm.step(x,1,x,prev,e))return 12;
 std::cout<<"Diffusion runtime math tests passed\n";return 0;}
