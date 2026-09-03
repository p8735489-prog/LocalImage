#pragma once
#include "../tensor/tensor.h"
#include "../diffusion/diffusion_runtime.h"
#include "../transformer/transformer_runtime.h"
#include "../safetensors/safe_tensor_file.h"
#include <string>
#include <vector>
#include <memory>

namespace localimage::sdxl {

struct SDXLConditioning {
    tensor::Tensor clip_l;       // [1,77,768]
    tensor::Tensor openclip_g;   // [1,77,1280]
    tensor::Tensor pooled_g;     // [1,1280]
    tensor::Tensor time_ids;     // [1,6]
};

struct SDXLRequest {
    std::string prompt;
    std::string negative_prompt;
    size_t steps=20;
    float cfg_scale=7.0f;
    uint64_t seed=0;
    size_t width=1024;
    size_t height=1024;
    diffusion::SchedulerType scheduler=diffusion::SchedulerType::Euler;
};

class SDXLTextEncoder {
public:
    bool encode(const std::vector<uint32_t>& ids, const safetensors::SafeTensorFile& weights,
                const std::string& prefix, tensor::Tensor& sequence, tensor::Tensor* pooled,
                std::string& error) const;
};

class SDXLUNet {
public:
    bool validateWeights(const safetensors::SafeTensorFile& weights, std::string& error) const;
    bool predict(const tensor::Tensor& latent, const SDXLConditioning& cond, double timestep,
                 const safetensors::SafeTensorFile& weights, tensor::Tensor& noise, std::string& error) const;
};

class SDXLVAE {
public:
    bool validateWeights(const safetensors::SafeTensorFile& weights, std::string& error) const;
    bool decode(const tensor::Tensor& latent, const safetensors::SafeTensorFile& weights,
                tensor::Tensor& rgb, std::string& error) const;
};

class SDXLRuntime {
public:
    bool load(const safetensors::SafeTensorFile& weights, std::string& error);
    bool generate(const SDXLRequest& request, tensor::Tensor& rgb, std::string& error);
    void unload();
    bool loaded() const { return weights_ != nullptr; }
private:
    const safetensors::SafeTensorFile* weights_=nullptr;
    SDXLTextEncoder text_;
    SDXLUNet unet_;
    SDXLVAE vae_;
};

} // namespace localimage::sdxl
