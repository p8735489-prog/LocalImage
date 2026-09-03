#include "component_catalog.h"

namespace localimage::models {
namespace {
bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
bool contains(const std::string& s, const char* p) { return s.find(p) != std::string::npos; }
}

const char* componentRoleName(ComponentRole r) {
    switch (r) {
        case ComponentRole::UNet: return "UNet";
        case ComponentRole::VAE: return "VAE";
        case ComponentRole::CLIP_L: return "CLIP-L";
        case ComponentRole::OpenCLIP_G: return "OpenCLIP-G";
        case ComponentRole::T5_XXL: return "T5-XXL";
        case ComponentRole::MMDiT: return "MMDiT";
        case ComponentRole::DiffusionTransformer: return "Diffusion Transformer";
        case ComponentRole::Tokenizer: return "Tokenizer";
        case ComponentRole::TextProjection: return "Text Projection";
        default: return "Unknown";
    }
}

ComponentRole classifyTensorName(Architecture a, const std::string& n) {
    if (starts(n, "first_stage_model.") || starts(n, "vae.") || starts(n, "decoder.") ||
        starts(n, "vae_decoder.") || contains(n, ".decoder.")) return ComponentRole::VAE;
    if (starts(n, "conditioner.embedders.0.") || starts(n, "text_encoder.") || starts(n, "clip_l.") || starts(n, "cond_stage_model.transformer.text_model.")) return ComponentRole::CLIP_L;
    if (starts(n, "conditioner.embedders.1.") || starts(n, "text_encoder_2.") ||
        starts(n, "clip_g.") || starts(n, "open_clip.") ||
        contains(n, "text_model.encoder.layers") && contains(n, "text_encoder_2")) return ComponentRole::OpenCLIP_G;
    if (starts(n, "text_encoder_3.") || starts(n, "t5xxl.") || starts(n, "t5.")) return ComponentRole::T5_XXL;
    if (starts(n, "model.diffusion_model.") || starts(n, "unet.") ||
        starts(n, "diffusion_model.") || starts(n, "model.diffusion_model")) return ComponentRole::UNet;
    if (a == Architecture::SD3 || a == Architecture::SD35) {
        if (contains(n, "joint_blocks.") || contains(n, "x_embedder.") || contains(n, "context_embedder.") || contains(n, "y_embedder.")) return ComponentRole::MMDiT;
    }
    if (a == Architecture::FLUX || a == Architecture::Anima) {
        if (contains(n, "double_blocks.") || contains(n, "single_blocks.") ||
            contains(n, "transformer_blocks.") || contains(n, "transformer.") ||
            contains(n, "img_in.") || contains(n, "txt_in.")) return ComponentRole::DiffusionTransformer;
    }
    if (starts(n, "text_projection") || contains(n, ".text_projection")) return ComponentRole::TextProjection;
    return ComponentRole::Unknown;
}

ComponentInventory inventory(const safetensors::SafeTensorFile& file, Architecture a) {
    ComponentInventory out;
    for (const auto& kv : file.tensors()) {
        const auto role = classifyTensorName(a, kv.first);
        switch (role) {
            case ComponentRole::UNet: out.unet.push_back(kv.first); break;
            case ComponentRole::VAE: out.vae.push_back(kv.first); break;
            case ComponentRole::CLIP_L: out.clipL.push_back(kv.first); break;
            case ComponentRole::OpenCLIP_G: out.openclipG.push_back(kv.first); break;
            case ComponentRole::T5_XXL: out.t5.push_back(kv.first); break;
            case ComponentRole::MMDiT:
            case ComponentRole::DiffusionTransformer: out.transformer.push_back(kv.first); break;
            default: out.unclassified.push_back(kv.first); break;
        }
    }
    return out;
}
} // namespace localimage::models
