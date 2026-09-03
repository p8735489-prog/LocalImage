#pragma once
#include "vulkan_context.h"
#include "vulkan_buffer.h"
#include "vulkan_shader.h"
#include "vulkan_pipeline.h"
#include "vulkan_tensor.h"
#include "../tensor/tensor.h"
#include "../runtime/ir/localimage_ir.h"
#include <string>
#include <vector>
namespace localimage::vulkan {
struct GpuTestResult{bool success=false;std::string report;double gpu_ms=-1,max_error=0;};
class VulkanCompute final {
    VulkanContext& c_;
    bool layout(VkDescriptorSetLayout&,uint32_t,std::string&);
    bool runElementwise(const tensor::Tensor&,const tensor::Tensor*,tensor::Tensor&,int,double&,std::string&);
    bool runGenerated(ir::Op,const std::vector<tensor::Tensor>&,const ir::Attributes&,tensor::Tensor&,double&,std::string&);
public:
    explicit VulkanCompute(VulkanContext&c):c_(c){}
    bool execute(ir::Op,const std::vector<tensor::Tensor>&,const ir::Attributes&,tensor::Tensor&,std::string&);
    // Executes directly on GPU-resident tensors. No CPU download/upload occurs between operators.
    bool executeResident(ir::Op,const std::vector<const VulkanTensor*>&,const ir::Attributes&,VulkanTensor&,std::string&);
    bool supported(std::string&)const;
    bool supportsOperator(ir::Op)const;
    GpuTestResult runTests();
};
}
