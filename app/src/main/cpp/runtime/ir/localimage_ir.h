#pragma once
#include "../../tensor/tensor.h"
#include <cstdint>
#include <string>
#include <vector>
namespace localimage::ir {
enum class Op { Input, Add, Sub, Mul, Div, Exp, Sqrt, Rsqrt, GELU, SiLU, Clamp, MatMul, BatchedMatMul, Linear, Conv2D, Softmax, LayerNorm, RMSNorm, GroupNorm, Reshape, Transpose, Slice, Concat, Broadcast, Upsample, Attention, RoPE };
struct TensorSpec { std::string name; tensor::TensorShape shape; tensor::TensorStride stride; tensor::TensorDType dtype=tensor::TensorDType::Unknown; bool contiguous=true; };
struct Attributes { std::vector<size_t> axes, permutation; std::vector<uint64_t> reshape_shape, broadcast_shape, slice_starts, slice_lengths; size_t groups=1,stride=1,padding=0,dilation=1,scale_factor=1; double epsilon=1e-5,attention_scale=0; bool keepdim=false; };
struct Value { uint32_t id=0; TensorSpec spec; };
struct Node { uint32_t id=0; Op op=Op::Input; std::vector<uint32_t> inputs,outputs; Attributes attr; std::string name; };
class Graph { public: uint32_t addValue(const TensorSpec&); uint32_t addNode(Op,std::vector<uint32_t>,std::string); bool setOutputs(uint32_t,std::vector<uint32_t>,std::string&); bool setAttributes(uint32_t,const Attributes&,std::string&); bool validate(std::string&) const; bool topological(std::vector<uint32_t>&,std::string&) const; const std::vector<Value>& values()const{return values_;} const std::vector<Node>& nodes()const{return nodes_;} private: std::vector<Value> values_; std::vector<Node> nodes_; };
const char* opName(Op);
}
