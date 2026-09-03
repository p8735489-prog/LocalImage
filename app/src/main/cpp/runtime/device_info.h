#pragma once

#include <cstdint>
#include <string>

namespace localimage::runtime {

struct DeviceInfo {
    std::string vendor_name;
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    std::string device_uuid_hex;
    uint32_t max_compute_workgroup_size_x=0,max_compute_workgroup_size_y=0,max_compute_workgroup_size_z=0;
    uint32_t max_compute_shared_memory=0; bool timestamp_compute_supported=false;
    uint64_t device_local_heap_bytes=0;

    std::string stableIdentity() const;
    std::string summary() const;
};

} // namespace localimage::runtime
