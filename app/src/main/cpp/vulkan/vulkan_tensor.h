#pragma once
#include "vulkan_context.h"
#include "vulkan_buffer.h"
#include "vulkan_memory.h"
#include "../tensor/tensor.h"
namespace localimage::vulkan {
class VulkanTensor final {
    VkDevice device_=VK_NULL_HANDLE; VkPhysicalDevice physical_=VK_NULL_HANDLE; uint32_t queueFamily_=VK_QUEUE_FAMILY_IGNORED;
    VkBuffer gpuBuffer_=VK_NULL_HANDLE; VkDeviceMemory gpuMemory_=VK_NULL_HANDLE; tensor::TensorShape shape_; tensor::TensorDType dtype_=tensor::TensorDType::Unknown;
public:
    ~VulkanTensor();
    VulkanTensor()=default;
    VulkanTensor(const VulkanTensor&)=delete;
    VulkanTensor& operator=(const VulkanTensor&)=delete;
    VulkanTensor(VulkanTensor&& other) noexcept;
    VulkanTensor& operator=(VulkanTensor&& other) noexcept;
    bool allocate(VkPhysicalDevice,VkDevice,uint32_t,const tensor::TensorShape&,tensor::TensorDType,std::string&);
    bool upload(VkPhysicalDevice,VkDevice,uint32_t,const tensor::Tensor&,VkQueue,std::string&);
    bool download(VkQueue,tensor::Tensor&,std::string&) const;
    void destroy();
    VkBuffer buffer()const{return gpuBuffer_;} VkDevice device()const{return device_;} VkPhysicalDevice physicalDevice()const{return physical_;} uint32_t queueFamily()const{return queueFamily_;} VkDeviceSize byteSize()const; const tensor::TensorShape& shape()const{return shape_;} tensor::TensorDType dtype()const{return dtype_;} bool resident()const{return gpuBuffer_!=VK_NULL_HANDLE;}
};
}
