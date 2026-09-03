#pragma once

#include "../models/model_detector.h"
#include "../tensor/tensor.h"
#include <cstdint>
#include <string>

namespace localimage::runtime {

enum class ResolutionBackend { Auto, CPU, Vulkan, NPU };

enum class ResolutionReason {
    NativeDefault,
    UserRequested,
    AlignedToVAE,
    LimitedByMemory,
    InvalidRequest
};

struct ModelResolutionInfo {
    models::Architecture architecture = models::Architecture::Unknown;
    uint32_t native_width = 512;
    uint32_t native_height = 512;
    uint32_t vae_downsample = 8;
    uint32_t latent_alignment = 1;
    uint32_t min_width = 64;
    uint32_t min_height = 64;
    uint32_t max_width = 0;
    uint32_t max_height = 0;
    // Approximate multiplier for peak working memory. It is deliberately
    // architecture-specific rather than a fixed per-resolution constant.
    double activation_multiplier = 1.0;
    uint64_t weight_bytes = 0;
};

struct ResolutionRequest {
    uint32_t width = 0;
    uint32_t height = 0;
    tensor::TensorDType dtype = tensor::TensorDType::F16;
    ResolutionBackend backend = ResolutionBackend::Auto;
    uint64_t available_cpu_bytes = 0;
    uint64_t available_gpu_bytes = 0;
    uint64_t available_npu_bytes = 0;
    bool user_specified = false;
};

struct ResolvedResolution {
    uint32_t requested_width = 0;
    uint32_t requested_height = 0;
    // width/height are the actual diffusion execution dimensions. They are
    // always legal for the model graph. output_width/output_height are the
    // requested/final image dimensions after VAE/UNet alignment. A final
    // resize may be required when the requested output is not a legal graph
    // shape (for example SDXL 1080x1080 -> execute 1088x1088 -> output 1080x1080).
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t latent_width = 0;
    uint32_t latent_height = 0;
    uint32_t vae_downsample = 0;
    uint32_t alignment = 0;
    uint64_t estimated_peak_bytes = 0;
    uint64_t memory_budget_bytes = 0;
    bool adjusted = false;
    bool memory_limited = false;
    ResolutionReason reason = ResolutionReason::NativeDefault;
    std::string warning;
};

class ResolutionPolicy final {
public:
    static ModelResolutionInfo defaults(models::Architecture architecture);
    static bool validate(const ModelResolutionInfo& model, uint32_t width, uint32_t height, std::string& error);
};

class ResolutionPlanner final {
public:
    bool resolve(const ModelResolutionInfo& model,
                 const ResolutionRequest& request,
                 ResolvedResolution& output,
                 std::string& error) const;
};

const char* resolutionReasonName(ResolutionReason reason);

} // namespace localimage::runtime
