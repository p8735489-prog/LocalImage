#pragma once

#include <cstdint>
#include <string>
#include <vulkan/vulkan.h>
#include "../runtime/device_info.h"

namespace localimage {

struct VulkanDeviceCapabilities {
    bool shader_float16 = false;
    bool storage_buffer_16bit = false;
    bool shader_int16 = false;
    bool subgroup = false;
    uint32_t subgroup_size = 0;
    uint32_t max_storage_buffer_range = 0;
    uint32_t max_push_constants_size = 0;
    uint32_t max_workgroup_x = 0;
    uint32_t max_workgroup_y = 0;
    uint32_t max_workgroup_z = 0;
    uint32_t shared_memory = 0;
};

class VulkanContext final {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool initialize(std::string& error);
    void shutdown();

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkQueue computeQueue() const { return compute_queue_; }
    uint32_t computeQueueFamily() const { return compute_queue_family_; }

    std::string deviceSummary() const;
    localimage::runtime::DeviceInfo deviceInfo() const;
    VulkanDeviceCapabilities capabilities() const;

private:
    bool createInstance(std::string& error);
    bool selectPhysicalDevice(std::string& error);
    bool createLogicalDevice(std::string& error);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = UINT32_MAX;
    uint32_t api_version_ = VK_API_VERSION_1_0;
};

} // namespace localimage
