#pragma once
#include "model_detector.h"
#include <cstdint>
#include <string>
#include <vector>

namespace localimage::models {

enum class ModelComponent : uint8_t {
    UNet,
    VAE,
    CLIP_L,
    OpenCLIP_G,
    T5_XXL,
    MMDiT,
    DiffusionTransformer,
    Tokenizer,
    Scheduler
};

struct ResolutionDefaults {
    uint32_t width = 512;
    uint32_t height = 512;
    uint32_t vaeDownsample = 8;
    uint32_t latentMultiple = 1;
};

struct ModelFamilySpec {
    Architecture architecture = Architecture::Unknown;
    const char* id = "unknown";
    const char* displayName = "Unknown";
    ResolutionDefaults resolution{};
    std::vector<ModelComponent> required;
    std::vector<ModelComponent> optional;
};

const ModelFamilySpec& modelFamilySpec(Architecture architecture);
const char* modelComponentName(ModelComponent component);

struct ComponentValidation {
    bool valid = false;
    std::vector<ModelComponent> missing;
    std::string error;
};

ComponentValidation validateComponents(Architecture architecture,
                                        const ComponentFlags& detected,
                                        bool tokenizerPresent = false,
                                        bool schedulerPresent = false);

bool isDiffusionTransformerFamily(Architecture architecture);

} // namespace localimage::models
