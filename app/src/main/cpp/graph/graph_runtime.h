#pragma once
#include "../tensor/tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace localimage { class VulkanContext; namespace vulkan { class VulkanCompute; } namespace graph {

enum class OpType { Input, Add, Sub, Mul, Div, Exp, Sqrt, Rsqrt, Clamp, MatMul, BatchedMatMul, Linear, Conv2D, GroupNorm, LayerNorm, RMSNorm, SiLU, GELU, Softmax, Reshape, Transpose, Slice, Concat, Broadcast, Upsample, Attention, RoPE };
const char* opName(OpType);

struct GraphValue { uint32_t id=0; std::string name; tensor::TensorShape shape; tensor::TensorDType dtype=tensor::TensorDType::Unknown; };
struct GraphNode {
 uint32_t id=0;
 OpType op=OpType::Input;
 std::vector<uint32_t> inputs;
 std::vector<uint32_t> outputs;
 std::string name;
 std::vector<size_t> axes;
 std::vector<size_t> permutation;
 size_t groups=1;
 size_t stride=1;
 size_t padding=0;
 size_t dilation=1;
 size_t scale_factor=1;
 double epsilon=1e-5;
 double attention_scale=0.0;
 bool keepdim=false;
 std::vector<uint64_t> reshape_shape;
 std::vector<uint64_t> broadcast_shape;
 std::vector<uint64_t> slice_starts;
 std::vector<uint64_t> slice_lengths;
};

class TensorRegistry;

class Graph {
public:
 uint32_t addValue(std::string name,const tensor::TensorShape& shape,tensor::TensorDType dtype);
 uint32_t addNode(OpType op,std::vector<uint32_t> inputs,std::string name);
 bool setNodeOutputs(uint32_t node,std::vector<uint32_t> outputs,std::string& error);
 bool setNodeAttributes(uint32_t node, std::vector<size_t> axes, std::vector<size_t> permutation,
                        size_t groups, size_t stride, size_t padding, size_t dilation,
                        size_t scale_factor, double epsilon, double attention_scale,
                        bool keepdim, std::string& error);
 bool setNodeShapeAttributes(uint32_t node, std::vector<uint64_t> reshapeShape,
                             std::vector<uint64_t> broadcastShape,
                             std::vector<uint64_t> sliceStarts,
                             std::vector<uint64_t> sliceLengths, std::string& error);
 bool validate(std::string& error) const;
 bool execute(const TensorRegistry& inputs, TensorRegistry& outputs,
              bool preferVulkan, bool preferNpu, std::string& backend, std::string& error) const;
 const std::vector<GraphValue>& values() const{return values_;}
 const std::vector<GraphNode>& nodes() const{return nodes_;}
 bool topologicalSort(std::vector<uint32_t>& order,std::string& error) const;
 bool inferNodeOutput(uint32_t nodeId, tensor::TensorShape& shape, tensor::TensorDType& dtype, std::string& error) const;
private: std::vector<GraphValue> values_; std::vector<GraphNode> nodes_;};

class TensorRegistry {
public:
 bool put(const std::string& name,const tensor::Tensor& t,std::string& error);
 bool get(const std::string& name,tensor::Tensor& out,std::string& error) const;
 bool has(const std::string& name) const;
 size_t size() const{return tensors_.size();}
private: std::unordered_map<std::string,tensor::Tensor> tensors_;};

struct ExecutionStep { uint32_t node_id=0; OpType op=OpType::Input; std::string name; };
class ExecutionPlanner {public: bool build(const Graph&,std::vector<ExecutionStep>&,std::string&) const;};

class CPUBackend {
public: bool execute(OpType op,const std::vector<tensor::Tensor>& inputs,tensor::Tensor& output,std::string& error, const GraphNode* node=nullptr) const;
};
class VulkanBackend {
public:
 VulkanBackend();
 ~VulkanBackend();
 VulkanBackend(const VulkanBackend&) = delete;
 VulkanBackend& operator=(const VulkanBackend&) = delete;
 bool available(std::string& error) const;
 bool execute(OpType op,const std::vector<tensor::Tensor>& inputs,tensor::Tensor& output,const GraphNode& node,std::string& error) const;
private:
#ifndef LOCALIMAGE_NO_VULKAN
 std::unique_ptr<localimage::VulkanContext> context_;
 std::unique_ptr<localimage::vulkan::VulkanCompute> compute_;
#endif
};

// NPU Backend: delegates to Qualcomm QNN HTP (Hexagon DSP)
// Follows the same pattern as VulkanBackend for easy integration.
class NpuBackend {
public:
 NpuBackend();
 ~NpuBackend();
 NpuBackend(const NpuBackend&) = delete;
 NpuBackend& operator=(const NpuBackend&) = delete;
 bool available(std::string& error) const;
 bool execute(OpType op,const std::vector<tensor::Tensor>& inputs,tensor::Tensor& output,const GraphNode& node,std::string& error) const;
private:
 struct Impl;
 std::unique_ptr<Impl> impl_;
};

} // namespace graph
} // namespace localimage
