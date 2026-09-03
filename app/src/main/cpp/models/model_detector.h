#pragma once
#include "../safetensors/safe_tensor_file.h"
#include <cstdint>
#include <string>
#include <vector>

namespace localimage::models {
enum class Architecture { StableDiffusion15, StableDiffusion2, SDXL, SD3, SD35, FLUX, Anima, Unknown };
struct ComponentFlags { bool unet=false, vae=false, clip=false, openclip=false, t5=false, transformer=false; };
struct Detection { Architecture architecture=Architecture::Unknown; bool supported=false; double confidence=0.0; std::string reason; ComponentFlags components; std::vector<std::string> requiredComponents; };
const char* architectureName(Architecture);
class ModelDetector { public: Detection detect(const safetensors::SafeTensorFile& file) const; };
}
