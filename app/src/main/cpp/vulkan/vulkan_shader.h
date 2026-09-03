#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <string>
namespace localimage::vulkan { class VulkanShader final {VkDevice d_=VK_NULL_HANDLE;VkShaderModule m_=VK_NULL_HANDLE;public:~VulkanShader();bool create(VkDevice,const uint32_t*,size_t,std::string&);void destroy();VkShaderModule handle()const{return m_;}};}
