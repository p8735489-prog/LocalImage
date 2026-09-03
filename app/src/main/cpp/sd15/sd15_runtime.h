#pragma once
#include "../graph/graph_runtime.h"
#include "../models/model_detector.h"
#include "../diffusion/diffusion_runtime.h"
#include <string>
namespace localimage::sd15 {
class SD15Runtime {
public:
 bool inspect(const safetensors::SafeTensorFile&, diffusion::DiffusionModelInfo&, std::string&) const;
 bool createScheduler(diffusion::SchedulerType, const diffusion::SchedulerConfig&, std::unique_ptr<diffusion::Scheduler>&, std::string&) const;
};
}
