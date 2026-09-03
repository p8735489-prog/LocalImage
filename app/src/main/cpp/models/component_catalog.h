#pragma once
#include "model_detector.h"
#include <string>
#include <vector>

namespace localimage::models {

enum class ComponentRole { Unknown, UNet, VAE, CLIP_L, OpenCLIP_G, T5_XXL, MMDiT, DiffusionTransformer, Tokenizer, TextProjection };

const char* componentRoleName(ComponentRole role);
ComponentRole classifyTensorName(Architecture architecture, const std::string& tensorName);

struct ComponentInventory {
    std::vector<std::string> unet;
    std::vector<std::string> vae;
    std::vector<std::string> clipL;
    std::vector<std::string> openclipG;
    std::vector<std::string> t5;
    std::vector<std::string> transformer;
    std::vector<std::string> unclassified;
};

ComponentInventory inventory(const safetensors::SafeTensorFile& file, Architecture architecture);

} // namespace localimage::models
