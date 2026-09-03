#pragma once
#include "../tensor/tensor.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace localimage::ops {

bool add(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, std::string&);
bool sub(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, std::string&);
bool mul(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, std::string&);
bool div(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, std::string&);

struct ErrorMetrics { double max_abs=0.0, mean_abs=0.0, max_rel=0.0; };

bool unary(const tensor::Tensor& x, tensor::Tensor& out, const std::string& op, std::string& error);
bool reduce(const tensor::Tensor& x, const std::vector<size_t>& axes, bool keepdim, tensor::Tensor& out, const std::string& op, std::string& error);
bool matmul(const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& out, std::string& error);
bool batchedMatmul(const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& out, std::string& error);
bool linear(const tensor::Tensor& x, const tensor::Tensor& w, const tensor::Tensor* bias, tensor::Tensor& out, std::string& error);
bool layerNorm(const tensor::Tensor& x, const tensor::Tensor& gamma, const tensor::Tensor* beta, double epsilon, tensor::Tensor& out, std::string& error);
bool groupNorm(const tensor::Tensor& x, const tensor::Tensor& gamma, const tensor::Tensor* beta, size_t groups, double epsilon, tensor::Tensor& out, std::string& error);
bool conv2d(const tensor::Tensor& x, const tensor::Tensor& weight, const tensor::Tensor* bias, size_t stride, size_t padding, size_t dilation, size_t groups, tensor::Tensor& out, std::string& error);
bool upsampleNearest(const tensor::Tensor& x, size_t scale, tensor::Tensor& out, std::string& error);
bool concat(const std::vector<tensor::Tensor>& xs, size_t axis, tensor::Tensor& out, std::string& error);
bool transpose(const tensor::Tensor& x, const std::vector<size_t>& perm, tensor::Tensor& out, std::string& error);
bool softmax(const tensor::Tensor& x, size_t axis, tensor::Tensor& out, std::string& error);
bool scaledDotProductAttention(const tensor::Tensor& q, const tensor::Tensor& k, const tensor::Tensor& v, double scale, const tensor::Tensor* mask, tensor::Tensor& out, std::string& error);
bool rmsNorm(const tensor::Tensor& x, const tensor::Tensor& gamma, double epsilon, tensor::Tensor& out, std::string& error);
bool rope(const tensor::Tensor& x, const tensor::Tensor& angles, tensor::Tensor& out, std::string& error);
bool broadcastTo(const tensor::Tensor& x, const tensor::TensorShape& shape, tensor::Tensor& out, std::string& error);

ErrorMetrics compare(const tensor::Tensor& a, const tensor::Tensor& b, std::string& error);

} // namespace localimage::ops
