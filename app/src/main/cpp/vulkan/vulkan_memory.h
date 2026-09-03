#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>
#include <string>
namespace localimage::vulkan {
class VulkanMemoryAllocator final {
public:
    VulkanMemoryAllocator(VkPhysicalDevice physical, VkDevice device, uint32_t queueFamily);
    bool createDeviceLocal(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory, std::string& error) const;
    bool createStaging(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, std::string& error) const;
    static void destroy(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory);
private:
    bool create(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory, std::string& error) const;
    bool findType(uint32_t bits, VkMemoryPropertyFlags properties, uint32_t& index, std::string& error) const;
    VkPhysicalDevice physical_=VK_NULL_HANDLE; VkDevice device_=VK_NULL_HANDLE; uint32_t queueFamily_=VK_QUEUE_FAMILY_IGNORED;
};
}
