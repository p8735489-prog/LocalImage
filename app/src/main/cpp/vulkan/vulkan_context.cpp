#include "vulkan_context.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <iomanip>
#include <vector>

namespace localimage {
namespace {

std::string resultString(VkResult result) {
    return std::to_string(static_cast<int>(result));
}

} // namespace

VulkanContext::~VulkanContext() {
    shutdown();
}

bool VulkanContext::initialize(std::string& error) {
    shutdown();
    if (!createInstance(error)) return false;
    if (!selectPhysicalDevice(error)) {
        shutdown();
        return false;
    }
    if (!createLogicalDevice(error)) {
        shutdown();
        return false;
    }
    return true;
}

void VulkanContext::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    compute_queue_ = VK_NULL_HANDLE;
    compute_queue_family_ = UINT32_MAX;
    physical_device_ = VK_NULL_HANDLE;

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    api_version_ = VK_API_VERSION_1_0;
}

bool VulkanContext::createInstance(std::string& error) {
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        const VkResult version_result = vkEnumerateInstanceVersion(&loader_version);
        if (version_result != VK_SUCCESS) {
            error = "vkEnumerateInstanceVersion failed: " + resultString(version_result);
            return false;
        }
    }

    api_version_ = std::min(loader_version, VK_API_VERSION_1_3);
    if (VK_API_VERSION_MINOR(api_version_) < 1) {
        error = "Vulkan 1.1 or newer is required; loader reports " +
                std::to_string(VK_API_VERSION_MAJOR(loader_version)) + "." +
                std::to_string(VK_API_VERSION_MINOR(loader_version));
        return false;
    }

    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "Local Image";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = "Local Image Native Runtime";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion = api_version_;

    VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &app_info;
    // Compute-only runtime: no surface extension is required. Keeping the instance minimal
    // also allows headless CPU/GPU execution on Android devices without a presentation stack.

    const VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        error = "vkCreateInstance failed: " + resultString(result);
        return false;
    }
    return true;
}

bool VulkanContext::selectPhysicalDevice(std::string& error) {
    uint32_t count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (result != VK_SUCCESS || count == 0) {
        error = "No Vulkan physical device found (result=" + resultString(result) + ")";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(count);
    result = vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    if (result != VK_SUCCESS) {
        error = "vkEnumeratePhysicalDevices failed: " + resultString(result);
        return false;
    }

    for (VkPhysicalDevice device : devices) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

        for (uint32_t i = 0; i < family_count; ++i) {
            const auto flags = families[i].queueFlags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) != 0 && families[i].queueCount > 0) {
                physical_device_ = device;
                compute_queue_family_ = i;
                return true;
            }
        }
    }

    error = "Vulkan device exists, but no compute-capable queue family was found";
    return false;
}

bool VulkanContext::createLogicalDevice(std::string& error) {
    constexpr float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = compute_queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    // Query first, then enable only features the selected device actually
    // advertises. Advertising a capability without enabling it would make a
    // later F16 shader path invalid.
    VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDevice16BitStorageFeatures available_storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    available.pNext = &available_storage;
    vkGetPhysicalDeviceFeatures2(physical_device_, &available);

    VkPhysicalDeviceFeatures2 enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDevice16BitStorageFeatures enabled_storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    enabled.pNext = &enabled_storage;
    enabled.features.shaderInt16 = available.features.shaderInt16;
    enabled_storage.storageBuffer16BitAccess = available_storage.storageBuffer16BitAccess;
    enabled_storage.uniformAndStorageBuffer16BitAccess = available_storage.uniformAndStorageBuffer16BitAccess;
    enabled_storage.storagePushConstant16 = available_storage.storagePushConstant16;

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pNext = &enabled;
    device_info.pEnabledFeatures = nullptr;

    const VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        error = "vkCreateDevice failed: " + resultString(result);
        return false;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    if (compute_queue_ == VK_NULL_HANDLE) {
        error = "vkGetDeviceQueue returned VK_NULL_HANDLE";
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

localimage::runtime::DeviceInfo VulkanContext::deviceInfo() const {
    localimage::runtime::DeviceInfo info;
    if (physical_device_ == VK_NULL_HANDLE) return info;

    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceIDProperties id_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    properties2.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    const VkPhysicalDeviceProperties& properties = properties2.properties;

    info.device_name = properties.deviceName;
    info.vendor_id = properties.vendorID;
    info.device_id = properties.deviceID;
    info.api_version = properties.apiVersion;
    info.driver_version = properties.driverVersion;
    info.max_compute_workgroup_size_x=properties.limits.maxComputeWorkGroupSize[0];
    info.max_compute_workgroup_size_y=properties.limits.maxComputeWorkGroupSize[1];
    info.max_compute_workgroup_size_z=properties.limits.maxComputeWorkGroupSize[2];
    info.max_compute_shared_memory=properties.limits.maxComputeSharedMemorySize;
    info.timestamp_compute_supported=properties.limits.timestampComputeAndGraphics!=0;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
        if ((memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            if (memory_properties.memoryHeaps[i].size > info.device_local_heap_bytes)
                info.device_local_heap_bytes = memory_properties.memoryHeaps[i].size;
        }
    }

    switch (properties.vendorID) {
        case 0x5143: info.vendor_name = "Qualcomm"; break;
        case 0x13B5: info.vendor_name = "ARM"; break;
        case 0x1002: info.vendor_name = "AMD"; break;
        case 0x10DE: info.vendor_name = "NVIDIA"; break;
        case 0x8086: info.vendor_name = "Intel"; break;
        default: info.vendor_name = "Unknown"; break;
    }

    if (std::any_of(std::begin(id_properties.deviceUUID), std::end(id_properties.deviceUUID), [](uint8_t byte) { return byte != 0; })) {
        std::ostringstream uuid;
        uuid << std::hex << std::setfill('0');
        for (uint8_t byte : id_properties.deviceUUID) uuid << std::setw(2) << static_cast<unsigned int>(byte);
        info.device_uuid_hex = uuid.str();
    }
    return info;
}

VulkanDeviceCapabilities VulkanContext::capabilities() const {
    VulkanDeviceCapabilities out;
    if (physical_device_ == VK_NULL_HANDLE) return out;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    features.pNext = &storage;
    vkGetPhysicalDeviceFeatures2(physical_device_, &features);
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    properties2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    const VkPhysicalDeviceProperties& properties = properties2.properties;
    out.shader_float16 = false;
    out.shader_int16 = features.features.shaderInt16 == VK_TRUE;
    out.storage_buffer_16bit = storage.storageBuffer16BitAccess == VK_TRUE;
    out.subgroup = subgroup.subgroupSize != 0;
    out.subgroup_size = subgroup.subgroupSize;
    out.max_storage_buffer_range = properties.limits.maxStorageBufferRange;
    out.max_push_constants_size = properties.limits.maxPushConstantsSize;
    out.max_workgroup_x = properties.limits.maxComputeWorkGroupSize[0];
    out.max_workgroup_y = properties.limits.maxComputeWorkGroupSize[1];
    out.max_workgroup_z = properties.limits.maxComputeWorkGroupSize[2];
    out.shared_memory = properties.limits.maxComputeSharedMemorySize;
    return out;
}

std::string VulkanContext::deviceSummary() const {
    if (physical_device_ == VK_NULL_HANDLE) return "No Vulkan device";

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);

    std::ostringstream out;
    out << "Vulkan "
        << VK_API_VERSION_MAJOR(properties.apiVersion) << "."
        << VK_API_VERSION_MINOR(properties.apiVersion) << "."
        << VK_API_VERSION_PATCH(properties.apiVersion)
        << "\nGPU: " << properties.deviceName
        << "\nVendor ID: 0x" << std::hex << properties.vendorID
        << "\nDevice ID: 0x" << properties.deviceID
        << std::dec
        << "\nCompute queue family: " << compute_queue_family_;
    return out.str();
}

} // namespace localimage
