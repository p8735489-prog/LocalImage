#pragma once

#include "device_info.h"

#include <string>

namespace localimage::runtime {

struct CacheKeyInput {
    std::string model_sha256;
    std::string model_version = "raw-safetensors";
    std::string runtime_version;
    std::string shader_version;
    DeviceInfo device;
};

class CacheKey {
public:
    static std::string build(const CacheKeyInput& input);
};

} // namespace localimage::runtime
