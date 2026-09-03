#pragma once
#include "../tensor/tensor.h"
#include "../safetensors/safe_tensor_file.h"
#include "../runtime/weight_store.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace localimage::diffusion {

enum class SchedulerType { DDIM, Euler, EulerAncestral, DPMPlusPlus2M, FlowMatch };

struct SchedulerConfig {
    SchedulerType type = SchedulerType::Euler;
    size_t inference_steps = 20;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    double prediction_scale = 1.0;
};

class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual bool configure(const SchedulerConfig&, std::string&) = 0;
    virtual bool step(const tensor::Tensor& model_output, double timestep,
                      const tensor::Tensor& sample, tensor::Tensor& previous,
                      std::string&) const = 0;
    virtual const std::vector<double>& timesteps() const = 0;
};

class DDIMScheduler final : public Scheduler {
public:
    bool configure(const SchedulerConfig&, std::string&) override;
    bool step(const tensor::Tensor&, double, const tensor::Tensor&, tensor::Tensor&, std::string&) const override;
    const std::vector<double>& timesteps() const override { return timesteps_; }
private:
    std::vector<double> timesteps_;
    std::vector<double> alpha_bars_;
};

class EulerScheduler final : public Scheduler {
public:
    bool configure(const SchedulerConfig&, std::string&) override;
    bool step(const tensor::Tensor&, double, const tensor::Tensor&, tensor::Tensor&, std::string&) const override;
    const std::vector<double>& timesteps() const override { return timesteps_; }
private:
    std::vector<double> timesteps_;
    std::vector<double> sigmas_;
};

class FlowMatchScheduler final : public Scheduler {
public:
    bool configure(const SchedulerConfig&, std::string&) override;
    bool step(const tensor::Tensor&, double, const tensor::Tensor&, tensor::Tensor&, std::string&) const override;
    const std::vector<double>& timesteps() const override { return timesteps_; }
private:
    std::vector<double> timesteps_;
};


struct ComponentInventory {
    bool unet = false;
    bool vae = false;
    bool clip = false;
    bool openclip = false;
    bool t5 = false;
    bool tokenizer = false;
    bool textProjection = false;
};

struct DiffusionModelInfo {
    std::string architecture;
    ComponentInventory components;
    size_t tensor_count = 0;
    uint64_t parameter_bytes = 0;
};

class DiffusionRuntime final {
public:
    bool inspect(const safetensors::SafeTensorFile&, DiffusionModelInfo&, std::string&) const;
    bool createScheduler(SchedulerType, const SchedulerConfig&, std::unique_ptr<Scheduler>&, std::string&) const;
};

bool rmsNorm(const tensor::Tensor&, const tensor::Tensor&, double, tensor::Tensor&, std::string&);
bool rotaryEmbedding(const tensor::Tensor&, const tensor::Tensor&, double, tensor::Tensor&, std::string&);
bool timestepEmbedding(const tensor::Tensor&, size_t, double, tensor::Tensor&, std::string&);

} // namespace localimage::diffusion
