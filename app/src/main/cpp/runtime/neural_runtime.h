#pragma once

#include "ir/localimage_ir.h"
#include "../tensor/tensor.h"
#include "../safetensors/safe_tensor_file.h"
#include "weight_store.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace localimage::runtime {

struct RuntimeResult {
    bool ok = false;
    std::string backend;
    std::string error;
};

class NeuralRuntime final {
public:
    bool loadWeights(const safetensors::SafeTensorFile& file, std::string& error);
    bool execute(const ir::Graph& graph,
                 const std::unordered_map<std::string, tensor::Tensor>& inputs,
                 std::unordered_map<std::string, tensor::Tensor>& outputs,
                 bool preferVulkan,
                 RuntimeResult& result) const;
    bool clear(std::string& error);
    bool loaded() const { return file_ != nullptr; }
private:
    const safetensors::SafeTensorFile* file_ = nullptr;
    WeightStore weights_;
};

} // namespace localimage::runtime
