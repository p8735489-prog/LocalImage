#pragma once
#include <vulkan/vulkan.h>
#include <string>
namespace localimage::vulkan { class VulkanBuffer final { VkDevice device_=VK_NULL_HANDLE; VkBuffer buffer_=VK_NULL_HANDLE; VkDeviceMemory memory_=VK_NULL_HANDLE; VkDeviceSize size_=0; public: ~VulkanBuffer(); bool create(VkPhysicalDevice,VkDevice,VkDeviceSize,VkBufferUsageFlags,VkMemoryPropertyFlags,uint32_t,std::string&); void destroy(); bool upload(const void*,VkDeviceSize,std::string&); bool download(void*,VkDeviceSize,std::string&) const; VkBuffer handle()const{return buffer_;} VkDeviceMemory memory()const{return memory_;} VkDeviceSize size()const{return size_;} private: bool findMemory(VkPhysicalDevice,uint32_t,VkMemoryPropertyFlags,uint32_t&,std::string&)const;}; }
