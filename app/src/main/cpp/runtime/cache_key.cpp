#include "cache_key.h"
#include "model_hash.h"

namespace localimage::runtime {

std::string CacheKey::build(const CacheKeyInput& input) {
    const std::string material =
        "model=" + input.model_sha256 +
        "\nmodel_version=" + input.model_version +
        "\nruntime=" + input.runtime_version +
        "\nshader=" + input.shader_version +
        "\ndevice=" + input.device.stableIdentity();

    ModelHash hash;
    std::string error;
    if (!hash.compute(reinterpret_cast<const uint8_t*>(material.data()), material.size(), error)) return {};
    return hash.hex();
}

} // namespace localimage::runtime
