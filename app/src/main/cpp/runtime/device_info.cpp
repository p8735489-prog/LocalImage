#include "device_info.h"

#include <iomanip>
#include <sstream>

namespace localimage::runtime {

std::string DeviceInfo::stableIdentity() const {
    std::ostringstream out;
    out << vendor_id << ':' << device_id << ':' << driver_version << ':' << api_version << ':' << device_uuid_hex;
    return out.str();
}

std::string DeviceInfo::summary() const {
    std::ostringstream out;
    out << "GPU: " << device_name
        << "\nVendor: " << vendor_name
        << "\nVendor ID: 0x" << std::hex << vendor_id
        << "\nDevice ID: 0x" << device_id
        << std::dec
        << "\nVulkan API: " << ((api_version >> 22) & 0x3ff) << '.' << ((api_version >> 12) & 0x3ff) << '.' << (api_version & 0xfff)
        << "\nDriver Version: " << driver_version
        << "\nDevice UUID: " << (device_uuid_hex.empty() ? "unavailable" : device_uuid_hex)
        << "\nCompute: " << max_compute_workgroup_size_x << "x" << max_compute_workgroup_size_y << "x" << max_compute_workgroup_size_z
        << "\nShared memory: " << max_compute_shared_memory
        << "\nDevice-local heap: " << device_local_heap_bytes << " bytes"
        << "\nTimestamp: " << (timestamp_compute_supported ? "supported" : "unavailable");
    return out.str();
}

} // namespace localimage::runtime
