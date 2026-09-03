#include "../tensor/tensor.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
using namespace localimage::tensor;
using localimage::safetensors::SafeTensorFile;
int main(){
 std::string e; TensorRuntime rt;
 Tensor a=rt.createTensor(TensorShape({2,3}),TensorDType::F32,e); assert(e.empty()&&a.valid()); float* p=(float*)a.mutableData(); for(int i=0;i<6;i++)p[i]=i+1;
 assert(a.shape().elementCount()==6&&a.stride().stride(0)==3&&a.stride().stride(1)==1);
 Tensor b=rt.createTensor(TensorShape({3}),TensorDType::F32,e); float* q=(float*)b.mutableData(); q[0]=10;q[1]=20;q[2]=30;
 Tensor c; assert(ops::add(a,b,c,e)); float ex[]={11,22,33,14,25,36}; for(int i=0;i<6;i++)assert(std::fabs(((float*)c.data())[i]-ex[i])<1e-6);
 Tensor r; assert(rt.reshape(a,TensorShape({3,2}),r,e)); assert(r.data()==a.data());
 Tensor s; assert(rt.slice(a,0,1,1,s,e)); assert(s.shape().dim(0)==1 && ((float*)s.data())[0]==4);
 assert(!TensorShape({UINT64_MAX,2}).valid());
 Tensor f16=rt.createTensor(TensorShape({2}),TensorDType::F16,e); assert(e.empty()&&f16.valid()); assert(f16.writeFloat32(0,1.5f,e)); assert(f16.writeFloat32(1,-2.25f,e)); assert(std::fabs(f16.readFloat32(0,e)-1.5f)<0.001f); assert(std::fabs(f16.readFloat32(1,e)+2.25f)<0.001f);
 const std::string h=R"({"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})"; char path[]="/tmp/li_st_XXXXXX"; int fd=mkstemp(path); assert(fd>=0); uint64_t n=h.size(); uint8_t le[8];for(int i=0;i<8;i++)le[i]=(n>>(8*i))&255; float v[]={1,2,3,4}; assert(write(fd,le,8)==8);assert(write(fd,h.data(),h.size())==(ssize_t)h.size());assert(write(fd,v,sizeof(v))==(ssize_t)sizeof(v));lseek(fd,0,SEEK_SET); SafeTensorFile sf;assert(sf.open(fd,e)&&sf.validate(e));localimage::safetensors::TensorView tv;assert(sf.getTensorView("x",tv,e));Tensor m=rt.createView(tv,e);assert(e.empty()&&m.device()==TensorDevice::MAPPED&&m.data()==tv.data());sf.close();assert(((float*)m.data())[2]==3);unlink(path);
 std::cout<<"Tensor Runtime host tests passed\n"; return 0;
}
