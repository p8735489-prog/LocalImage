#include "neural_runtime.h"
#include "hardware_planner.h"
#include "../operators/cpu_operators.h"
#include "../diffusion/diffusion_runtime.h"
#include "../vulkan/vulkan_context.h"
#include "../vulkan/vulkan_compute.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace localimage::runtime {
namespace {
using tensor::Tensor;
using tensor::TensorDType;

bool checkSpec(const ir::TensorSpec& spec, const Tensor& t, std::string& e) {
    if (!t.valid()) { e = "runtime received invalid tensor: " + spec.name; return false; }
    if (t.dtype() != spec.dtype) { e = "dtype mismatch for " + spec.name; return false; }
    if (t.shape().dims() != spec.shape.dims()) { e = "shape mismatch for " + spec.name; return false; }
    if (!spec.contiguous && t.isContiguous()) return true;
    if (spec.contiguous && !t.isContiguous()) { e = "contiguous tensor required for " + spec.name; return false; }
    return true;
}

bool executeCpu(const ir::Node& n, const std::vector<Tensor>& in, Tensor& out, std::string& e) {
    std::vector<Tensor> f32in;
    f32in.reserve(in.size());
    TensorRuntime rt;
    for (const auto& input : in) {
        if (!input.valid()) { e = "CPU operator received an invalid tensor"; return false; }
        Tensor x;
        if (!rt.convertDtype(input, TensorDType::F32, x, e)) return false;
        if (!x.isContiguous()) { e = "CPU materialization produced a non-contiguous tensor"; return false; }
        f32in.push_back(std::move(x));
    }
    const auto& in32 = f32in;
    using namespace localimage::ops;
    switch (n.op) {
        case ir::Op::Add: return add(in32[0], in32[1], out, e);
        case ir::Op::Sub: return sub(in32[0], in32[1], out, e);
        case ir::Op::Mul: return mul(in32[0], in32[1], out, e);
        case ir::Op::Div: return div(in32[0], in32[1], out, e);
        case ir::Op::Exp: return unary(in32[0], out, "exp", e);
        case ir::Op::Sqrt: return unary(in32[0], out, "sqrt", e);
        case ir::Op::Rsqrt: return unary(in32[0], out, "rsqrt", e);
        case ir::Op::GELU: return unary(in32[0], out, "gelu", e);
        case ir::Op::SiLU: return unary(in32[0], out, "silu", e);
        case ir::Op::Clamp: return unary(in32[0], out, "clamp", e);
        case ir::Op::MatMul: return matmul(in32[0], in32[1], out, e);
        case ir::Op::BatchedMatMul: return batchedMatmul(in32[0], in32[1], out, e);
        case ir::Op::Linear: return linear(in32[0], in32[1], in32.size() == 3 ? &in32[2] : nullptr, out, e);
        case ir::Op::Conv2D: return conv2d(in32[0], in32[1], in32.size() == 3 ? &in32[2] : nullptr, n.attr.stride, n.attr.padding, n.attr.dilation, n.attr.groups, out, e);
        case ir::Op::Softmax: return softmax(in32[0], n.attr.axes.empty() ? (in32[0].shape().rank() - 1) : n.attr.axes[0], out, e);
        case ir::Op::LayerNorm: return layerNorm(in32[0], in32[1], in32.size() == 3 ? &in32[2] : nullptr, n.attr.epsilon, out, e);
        case ir::Op::RMSNorm: return diffusion::rmsNorm(in32[0], in32[1], n.attr.epsilon, out, e);
        case ir::Op::GroupNorm: return groupNorm(in32[0], in32[1], in32.size() == 3 ? &in32[2] : nullptr, n.attr.groups, n.attr.epsilon, out, e);
        case ir::Op::Transpose: return tensor::TensorRuntime().transpose(in32[0], n.attr.permutation, out, e);
        case ir::Op::Slice: {
            if (n.attr.axes.size() != 3) { e = "Slice requires axes={dimension,start,length}"; return false; }
            return tensor::TensorRuntime().slice(in32[0], n.attr.axes[0], n.attr.reshape_shape.empty() ? 0 : n.attr.reshape_shape[0], n.attr.reshape_shape.size() > 1 ? n.attr.reshape_shape[1] : 0, out, e);
        }
        case ir::Op::Concat: return concat(in32, n.attr.axes.empty() ? 0 : n.attr.axes[0], out, e);
        case ir::Op::Upsample: return upsampleNearest(in32[0], n.attr.scale_factor, out, e);
        case ir::Op::Attention: return scaledDotProductAttention(in32[0], in32[1], in32[2], n.attr.attention_scale, in32.size() == 4 ? &in32[3] : nullptr, out, e);
        case ir::Op::RoPE: return diffusion::rotaryEmbedding(in32[0], in32[1], n.attr.attention_scale == 0.0 ? 1.0 : n.attr.attention_scale, out, e);
        case ir::Op::Reshape: return tensor::TensorRuntime().reshape(in32[0], tensor::TensorShape(n.attr.reshape_shape), out, e);
        case ir::Op::Broadcast: {
            e = "Broadcast is represented by materialization in the CPU reference; explicit broadcast shape is required";
            if (n.attr.broadcast_shape.empty()) return false;
            tensor::TensorShape target(n.attr.broadcast_shape);
            if (!target.valid()) { e = target.error(); return false; }
            // Materialize through Add(x, zero) to reuse the validated broadcast implementation.
            Tensor zero = tensor::TensorRuntime().createTensor(target, TensorDType::F32, e);
            if (!zero.valid()) return false;
            std::fill(static_cast<float*>(zero.mutableData()), static_cast<float*>(zero.mutableData()) + zero.shape().elementCount(), 0.0f);
            return add(in32[0], zero, out, e);
        }
        case ir::Op::Input: e = "Input cannot be executed as an operator"; return false;
    }
    e = "unsupported IR operator";
    return false;
}
}

bool NeuralRuntime::loadWeights(const safetensors::SafeTensorFile& file, std::string& error) {
    if (file.tensors().empty()) { error = "cannot load an empty SafeTensors model"; return false; }
    std::string validation;
    if (!const_cast<safetensors::SafeTensorFile&>(file).validate(validation)) { error = validation; return false; }
    if (!weights_.attach(file, error)) return false;
    file_ = &file;
    return true;
}

bool NeuralRuntime::clear(std::string& error) { file_ = nullptr; weights_.clear(); error.clear(); return true; }

bool NeuralRuntime::execute(const ir::Graph& graph,
                            const std::unordered_map<std::string, Tensor>& inputs,
                            std::unordered_map<std::string, Tensor>& outputs,
                            bool preferVulkan,
                            RuntimeResult& result) const {
    outputs.clear(); result = {};
    std::string e;
    if (!graph.validate(e)) { result.error = e; return false; }

    VulkanContext vk;
    std::unique_ptr<vulkan::VulkanCompute> vc;
    bool vkReady = false;
    if (preferVulkan && vk.initialize(e)) {
        vc = std::make_unique<vulkan::VulkanCompute>(vk);
        std::string ve;
        vkReady = vc->supported(ve);
    }

    std::vector<uint32_t> order;
    if (!graph.topological(order, e)) { result.error = e; return false; }
    std::unordered_map<uint32_t, Tensor> values;
    // GPU values are kept resident across the operators that explicitly support
    // resident execution. A CPU tensor is materialized only when a CPU boundary
    // is actually required by the execution plan or final output.
    std::unordered_map<uint32_t, std::unique_ptr<vulkan::VulkanTensor>> gpuValues;
    auto residentEligible = [](ir::Op op) {
        switch (op) {
            case ir::Op::Add: case ir::Op::Sub: case ir::Op::Mul: case ir::Op::Div:
            case ir::Op::Exp: case ir::Op::Sqrt: case ir::Op::Rsqrt: case ir::Op::Clamp:
            case ir::Op::SiLU: case ir::Op::MatMul: case ir::Op::BatchedMatMul: case ir::Op::Linear:
                return true;
            default: return false;
        }
    };
    auto downloadGpu = [&](uint32_t valueId, Tensor& dst) -> bool {
        auto it = gpuValues.find(valueId);
        if (it == gpuValues.end()) return false;
        return it->second->download(vk.computeQueue(), dst, result.error);
    };

    for (uint32_t id : order) {
        const auto& n = graph.nodes()[id];
        if (n.op == ir::Op::Input) {
            if (n.outputs.size() != 1) { result.error = "Input node must have one output"; return false; }
            const auto& spec = graph.values()[n.outputs[0]].spec;
            auto it = inputs.find(spec.name);
            if (it == inputs.end()) { result.error = "missing graph input: " + spec.name; return false; }
            if (!checkSpec(spec, it->second, result.error)) return false;
            values[n.outputs[0]] = it->second;
            if (vkReady) {
                auto gt = std::make_unique<vulkan::VulkanTensor>();
                if (!gt->upload(vk.physicalDevice(), vk.device(), vk.computeQueueFamily(), it->second, vk.computeQueue(), result.error)) {
                    result.backend = "Vulkan";
                    return false;
                }
                gpuValues[n.outputs[0]] = std::move(gt);
            }
            continue;
        }
        if (n.outputs.size() != 1) { result.error = "execution currently requires one output per node: " + n.name; return false; }
        Tensor out;
        bool done = false;
        bool residentDone = false;
        std::vector<Tensor> ins;
        std::string ve;
        const bool canResident = vkReady && vc && residentEligible(n.op);

        if (canResident) {
            std::vector<const vulkan::VulkanTensor*> residentInputs;
            residentInputs.reserve(n.inputs.size());
            bool prepared = true;
            for (uint32_t v : n.inputs) {
                auto git = gpuValues.find(v);
                if (git == gpuValues.end()) {
                    auto vit = values.find(v);
                    if (vit == values.end()) { prepared = false; break; }
                    auto gt = std::make_unique<vulkan::VulkanTensor>();
                    if (!gt->upload(vk.physicalDevice(), vk.device(), vk.computeQueueFamily(), vit->second, vk.computeQueue(), ve)) {
                        result.backend = "Vulkan";
                        result.error = ve.empty() ? "Vulkan input upload failed" : ve;
                        return false;
                    }
                    gpuValues[v] = std::move(gt);
                    git = gpuValues.find(v);
                }
                residentInputs.push_back(git->second.get());
            }
            if (prepared) {
                auto gout = std::make_unique<vulkan::VulkanTensor>();
                if (vc->executeResident(n.op, residentInputs, n.attr, *gout, ve)) {
                    gpuValues[n.outputs[0]] = std::move(gout);
                    done = true;
                    residentDone = true;
                    result.backend = "Vulkan (resident)";
                } else if (!ve.empty() &&
                           ve.rfind("Vulkan operator unavailable:", 0) != 0 &&
                           ve.rfind("Vulkan resident operator unavailable:", 0) != 0 &&
                           ve.rfind("Vulkan operator unavailable", 0) != 0) {
                    result.backend = "Vulkan";
                    result.error = ve;
                    return false;
                }
            }
        }

        if (!done) {
            // Materialize GPU values only at a real CPU/Vulkan-transfer boundary.
            // A resident chain never enters this path and therefore never performs
            // GPU -> CPU -> GPU between adjacent resident operators.
            ins.reserve(n.inputs.size());
            for (uint32_t v : n.inputs) {
                auto it = values.find(v);
                if (it != values.end()) { ins.push_back(it->second); continue; }
                Tensor materialized;
                if (!downloadGpu(v, materialized)) {
                    result.error = "unresolved input value " + std::to_string(v) + " at node " + n.name;
                    return false;
                }
                values[v] = materialized;
                ins.push_back(materialized);
            }

            if (vkReady && vc) {
                done = vc->execute(n.op, ins, n.attr, out, ve);
                if (done) {
                    result.backend = "Vulkan";
                } else {
                    const bool operatorUnavailable =
                        ve.rfind("Vulkan operator unavailable:", 0) == 0 ||
                        ve.rfind("Vulkan backend unavailable", 0) == 0 ||
                        ve.rfind("Vulkan device unavailable", 0) == 0;
                    if (!operatorUnavailable) {
                        result.backend = "Vulkan";
                        result.error = ve.empty() ? "Vulkan execution failed" : ve;
                        return false;
                    }
                }
            }
            if (!done) {
                if (!executeCpu(n, ins, out, e)) {
                    result.backend = vkReady ? "CPU (Vulkan operator unsupported)" : "CPU (Vulkan unavailable)";
                    result.error = e.empty() ? ve : e;
                    return false;
                }
                const auto& outputSpec = graph.values()[n.outputs[0]].spec;
                if (out.dtype() != outputSpec.dtype) {
                    Tensor converted;
                    if (!TensorRuntime().convertDtype(out, outputSpec.dtype, converted, e)) {
                        result.backend = "CPU";
                        result.error = "CPU output dtype conversion failed: " + e;
                        return false;
                    }
                    out = std::move(converted);
                }
                result.backend = "CPU";
            }
        }

        if (residentDone) continue;
        const auto& spec = graph.values()[n.outputs[0]].spec;
        if (!checkSpec(spec, out, e)) { result.error = e; return false; }
        values[n.outputs[0]] = out;
        // Keep a GPU copy only when a later resident operator actually consumes it.
        if (vkReady && !gpuValues.count(n.outputs[0])) {
            bool futureResidentUse = false;
            for (size_t oi = 0; oi < order.size(); ++oi) {
                if (order[oi] != id) continue;
                for (size_t oj = oi + 1; oj < order.size(); ++oj) {
                    const auto& later = graph.nodes()[order[oj]];
                    if (!residentEligible(later.op)) continue;
                    if (std::find(later.inputs.begin(), later.inputs.end(), n.outputs[0]) != later.inputs.end()) {
                        futureResidentUse = true; break;
                    }
                }
                break;
            }
            if (futureResidentUse) {
                auto gt = std::make_unique<vulkan::VulkanTensor>();
                if (!gt->upload(vk.physicalDevice(), vk.device(), vk.computeQueueFamily(), out, vk.computeQueue(), e)) {
                    result.backend = "Vulkan";
                    result.error = e;
                    return false;
                }
                gpuValues[n.outputs[0]] = std::move(gt);
            }
        }
    }

    for (const auto& v : graph.values()) {
        auto it = values.find(v.id);
        if (it != values.end()) { outputs[v.spec.name] = it->second; continue; }
        auto git = gpuValues.find(v.id);
        if (git != gpuValues.end()) {
            Tensor materialized;
            if (!git->second->download(vk.computeQueue(), materialized, result.error)) { result.backend = "Vulkan"; return false; }
            if (!checkSpec(v.spec, materialized, result.error)) { result.backend = "Vulkan"; return false; }
            outputs[v.spec.name] = materialized;
        }
    }
    result.ok = true;
    result.error.clear();
    return true;
}

} // namespace localimage::runtime
