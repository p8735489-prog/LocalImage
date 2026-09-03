#pragma once
#include "../tensor/tensor.h"
#include <string>
namespace localimage::transformer {
using tensor::Tensor;
struct TransformerConfig { size_t hidden=0, heads=0, intermediate=0; double eps=1e-5; };
bool rmsNorm(const Tensor&,const Tensor&,double,Tensor&,std::string&);
bool linear(const Tensor&,const Tensor&,const Tensor*,Tensor&,std::string&);
bool gelu(const Tensor&,Tensor&,std::string&);
bool selfAttention(const Tensor&,const Tensor&,const Tensor&,double,const Tensor*,Tensor&,std::string&);
bool crossAttention(const Tensor&,const Tensor&,const Tensor&,double,const Tensor*,Tensor&,std::string&);
bool rope(const Tensor&,const Tensor&,Tensor&,std::string&);
}
