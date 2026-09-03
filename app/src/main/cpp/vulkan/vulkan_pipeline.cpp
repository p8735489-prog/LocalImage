#include "vulkan_pipeline.h"
namespace localimage::vulkan {
VulkanPipeline::~VulkanPipeline(){destroy();}
bool VulkanPipeline::createCompute(VkDevice d,VkShaderModule s,VkDescriptorSetLayout dl,std::string&e){return createCompute(d,s,dl,0,e);}
bool VulkanPipeline::createCompute(VkDevice d,VkShaderModule s,VkDescriptorSetLayout dl,uint32_t bytes,std::string&e){destroy();d_=d;VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&dl;VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,bytes};if(bytes){li.pushConstantRangeCount=1;li.pPushConstantRanges=&range;}VkResult r=vkCreatePipelineLayout(d,&li,nullptr,&l_);if(r!=VK_SUCCESS){e="vkCreatePipelineLayout: "+std::to_string((int)r);destroy();return false;}VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=VK_SHADER_STAGE_COMPUTE_BIT;si.module=s;si.pName="main";VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage=si;ci.layout=l_;r=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&ci,nullptr,&p_);if(r!=VK_SUCCESS){e="vkCreateComputePipelines: "+std::to_string((int)r);destroy();return false;}return true;}
void VulkanPipeline::destroy(){if(d_){if(p_)vkDestroyPipeline(d_,p_,nullptr);if(l_)vkDestroyPipelineLayout(d_,l_,nullptr);}d_=VK_NULL_HANDLE;p_=VK_NULL_HANDLE;l_=VK_NULL_HANDLE;}
}
