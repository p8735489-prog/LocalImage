#include "model_family.h"
#include <algorithm>

namespace localimage::models {
namespace {
const ModelFamilySpec UNKNOWN{Architecture::Unknown, "unknown", "Unknown", {512,512,8,1}, {}, {}};
const ModelFamilySpec SD15{Architecture::StableDiffusion15, "sd15", "Stable Diffusion 1.x", {512,512,8,1},
    {ModelComponent::UNet,ModelComponent::VAE,ModelComponent::CLIP_L}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec SD2{Architecture::StableDiffusion2, "sd2", "Stable Diffusion 2.x", {768,768,8,1},
    {ModelComponent::UNet,ModelComponent::VAE,ModelComponent::OpenCLIP_G}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec XL{Architecture::SDXL, "sdxl", "Stable Diffusion XL", {1024,1024,8,1},
    {ModelComponent::UNet,ModelComponent::VAE,ModelComponent::CLIP_L,ModelComponent::OpenCLIP_G}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec SD3{Architecture::SD3, "sd3", "Stable Diffusion 3", {1080,1080,8,1},
    {ModelComponent::VAE,ModelComponent::CLIP_L,ModelComponent::T5_XXL,ModelComponent::MMDiT}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec SD35{Architecture::SD35, "sd35", "Stable Diffusion 3.5", {1080,1080,8,1},
    {ModelComponent::VAE,ModelComponent::CLIP_L,ModelComponent::T5_XXL,ModelComponent::MMDiT}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec FLUX{Architecture::FLUX, "flux", "FLUX", {1080,1080,8,1},
    {ModelComponent::VAE,ModelComponent::CLIP_L,ModelComponent::T5_XXL,ModelComponent::DiffusionTransformer}, {ModelComponent::Tokenizer,ModelComponent::Scheduler}};
const ModelFamilySpec ANIMA{Architecture::Anima, "anima", "Anima", {1024,1024,8,1},
    {ModelComponent::VAE,ModelComponent::DiffusionTransformer}, {ModelComponent::CLIP_L,ModelComponent::T5_XXL,ModelComponent::Tokenizer,ModelComponent::Scheduler}};

bool has(const ComponentFlags& c, ModelComponent x) {
    switch (x) {
        case ModelComponent::UNet: return c.unet;
        case ModelComponent::VAE: return c.vae;
        case ModelComponent::CLIP_L: return c.clip;
        case ModelComponent::OpenCLIP_G: return c.openclip;
        case ModelComponent::T5_XXL: return c.t5;
        case ModelComponent::MMDiT: return c.transformer;
        case ModelComponent::DiffusionTransformer: return c.transformer;
        case ModelComponent::Tokenizer: return false;
        case ModelComponent::Scheduler: return false;
    }
    return false;
}
}

const ModelFamilySpec& modelFamilySpec(Architecture a) {
    switch (a) {
        case Architecture::StableDiffusion15: return SD15;
        case Architecture::StableDiffusion2: return SD2;
        case Architecture::SDXL: return XL;
        case Architecture::SD3: return SD3;
        case Architecture::SD35: return SD35;
        case Architecture::FLUX: return FLUX;
        case Architecture::Anima: return ANIMA;
        default: return UNKNOWN;
    }
}

const char* modelComponentName(ModelComponent c) {
    switch (c) {
        case ModelComponent::UNet: return "UNet";
        case ModelComponent::VAE: return "VAE";
        case ModelComponent::CLIP_L: return "CLIP-L";
        case ModelComponent::OpenCLIP_G: return "OpenCLIP-G";
        case ModelComponent::T5_XXL: return "T5-XXL";
        case ModelComponent::MMDiT: return "MMDiT";
        case ModelComponent::DiffusionTransformer: return "Diffusion Transformer";
        case ModelComponent::Tokenizer: return "Tokenizer";
        case ModelComponent::Scheduler: return "Scheduler";
    }
    return "Unknown";
}

ComponentValidation validateComponents(Architecture architecture, const ComponentFlags& detected,
                                       bool tokenizerPresent, bool schedulerPresent) {
    ComponentValidation out;
    const auto& spec = modelFamilySpec(architecture);
    if (architecture == Architecture::Unknown) { out.error = "unknown model architecture"; return out; }
    for (auto c : spec.required) if (!has(detected, c)) out.missing.push_back(c);
    for (auto c : spec.optional) {
        if (c == ModelComponent::Tokenizer && tokenizerPresent) continue;
        if (c == ModelComponent::Scheduler && schedulerPresent) continue;
    }
    out.valid = out.missing.empty();
    if (!out.valid) {
        out.error = "missing required components:";
        for (size_t i=0;i<out.missing.size();++i) {
            out.error += (i ? ", " : " ");
            out.error += modelComponentName(out.missing[i]);
        }
    }
    return out;
}

bool isDiffusionTransformerFamily(Architecture a) {
    return a == Architecture::SD3 || a == Architecture::SD35 || a == Architecture::FLUX || a == Architecture::Anima;
}

} // namespace localimage::models
