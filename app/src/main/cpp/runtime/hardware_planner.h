#pragma once
#include "ir/localimage_ir.h"
#include "../npu/npu_backend.h"
#include <string>
#include <vector>
namespace localimage::runtime {
enum class BackendKind { CPU, Vulkan, NPU };
struct DeviceCapabilities { bool vulkan=false,npu=false; std::string gpu; std::vector<std::string> vulkanOps; std::vector<std::string> npuOps; };
struct ExecutionAssignment { uint32_t node=0; BackendKind backend=BackendKind::CPU; };
struct ExecutionPlan { std::vector<ExecutionAssignment> steps; };
class HardwarePlanner { public: bool build(const ir::Graph&,const DeviceCapabilities&,bool preferNpu,ExecutionPlan&,std::string&) const; };
}
