#pragma once
#include "model_detector.h"
#include "model_family.h"
#include "component_catalog.h"
#include <string>

namespace localimage::models {

struct PackageInspection {
    Detection detection{};
    ComponentValidation components{};
    uint64_t tensorCount = 0;
    uint64_t parameterBytes = 0;
    bool executable = false;
    bool conversionReady = false;
    bool tokenizerEmbedded = false;
    bool schedulerEmbedded = false;
    ComponentInventory inventory{};
    std::string error;
};

class ModelPackageInspector final {
public:
    bool inspect(const safetensors::SafeTensorFile& file, PackageInspection& output, std::string& error) const;
};

} // namespace localimage::models
