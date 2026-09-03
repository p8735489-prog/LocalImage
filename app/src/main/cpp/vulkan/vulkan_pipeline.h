#pragma once
#include <vulkan/vulkan.h>
#include <string>
namespace localimage::vulkan {
class VulkanPipeline final {
    VkDevice d_=VK_NULL_HANDLE; VkPipeline p_=VK_NULL_HANDLE; VkPipelineLayout l_=VK_NULL_HANDLE;
public:
    ~VulkanPipeline();
    bool createCompute(VkDevice,VkShaderModule,VkDescriptorSetLayout,std::string&);
    bool createCompute(VkDevice,VkShaderModule,VkDescriptorSetLayout,uint32_t pushConstantBytes,std::string&);
    void destroy(); VkPipeline handle()const{return p_;} VkPipelineLayout layout()const{return l_;}
};
}
