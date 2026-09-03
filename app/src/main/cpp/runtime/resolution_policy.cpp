#include "resolution_policy.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace localimage::runtime {
namespace {
uint64_t safeMul(uint64_t a, uint64_t b, bool& ok) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) { ok = false; return 0; }
    return a * b;
}
uint64_t dtypeBytes(tensor::TensorDType d) {
    const size_t n = tensor::dtypeSize(d);
    return n == 0 ? 0 : static_cast<uint64_t>(n);
}
uint64_t budgetFor(const ResolutionRequest& r) {
    uint64_t raw = 0;
    switch (r.backend) {
        case ResolutionBackend::CPU: raw = r.available_cpu_bytes; break;
        case ResolutionBackend::Vulkan: raw = r.available_gpu_bytes; break;
        case ResolutionBackend::NPU: raw = r.available_npu_bytes; break;
        case ResolutionBackend::Auto:
            if (r.available_gpu_bytes) raw = r.available_gpu_bytes;
            else if (r.available_npu_bytes) raw = r.available_npu_bytes;
            else raw = r.available_cpu_bytes;
            break;
    }
    if (!raw) return 0;
    // Keep allocator headroom for the OS, driver and transient allocations.
    return static_cast<uint64_t>(static_cast<long double>(raw) * 0.80L);
}
uint64_t estimatePeak(const ModelResolutionInfo& m, uint32_t w, uint32_t h,
                       tensor::TensorDType dtype, bool& ok) {
    const uint64_t elemBytes = dtypeBytes(dtype);
    if (!elemBytes || w == 0 || h == 0 || m.vae_downsample == 0) {
        ok = false; return 0;
    }

    const uint64_t lw = w / m.vae_downsample;
    const uint64_t lh = h / m.vae_downsample;
    uint64_t latentArea = safeMul(lw, lh, ok);
    if (!ok) return 0;

    // The estimate intentionally includes the major resident/working classes:
    // latent (4 channels), UNet/DiT activations, attention workspace and VAE.
    // Weight bytes are supplied by the model inspector when known.
    uint64_t latent = safeMul(latentArea, 4, ok);
    latent = safeMul(latent, elemBytes, ok);
    if (!ok) return 0;

    const uint64_t activationUnits =
        static_cast<uint64_t>(std::ceil(32.0 * std::max(1.0, m.activation_multiplier)));
    uint64_t activations = safeMul(latent, activationUnits, ok);
    if (!ok) return 0;

    uint64_t vae = safeMul(latentArea, 16, ok);
    vae = safeMul(vae, elemBytes, ok);
    if (!ok) return 0;

    // Attention memory grows with token/spatial area. This is a bounded
    // planner estimate, not a claim about the exact backend allocator.
    uint64_t attention = safeMul(latentArea, 24, ok);
    attention = safeMul(attention, elemBytes, ok);
    if (!ok) return 0;

    uint64_t working = 0;
    if (activations > std::numeric_limits<uint64_t>::max() - vae) { ok = false; return 0; }
    working = activations + vae;
    if (working > std::numeric_limits<uint64_t>::max() - attention) { ok = false; return 0; }
    working += attention;
    if (m.weight_bytes > std::numeric_limits<uint64_t>::max() - working) { ok = false; return 0; }
    return working + m.weight_bytes;
}

uint32_t alignDown(uint32_t value, uint32_t alignment) {
    return alignment == 0 ? 0 : value - (value % alignment);
}
uint32_t alignNearest(uint32_t value, uint32_t alignment) {
    if (alignment == 0) return 0;
    const uint32_t down = alignDown(value, alignment);
    const uint64_t up64 = static_cast<uint64_t>(down) + alignment;
    if (up64 > std::numeric_limits<uint32_t>::max()) return down;
    const uint32_t up = static_cast<uint32_t>(up64);
    return (static_cast<uint64_t>(value) - down) < (static_cast<uint64_t>(up) - value) ? down : up;
}
}

const char* resolutionReasonName(ResolutionReason reason) {
    switch (reason) {
        case ResolutionReason::NativeDefault: return "native-default";
        case ResolutionReason::UserRequested: return "user-requested";
        case ResolutionReason::AlignedToVAE: return "aligned-to-vae";
        case ResolutionReason::LimitedByMemory: return "limited-by-memory";
        case ResolutionReason::InvalidRequest: return "invalid-request";
    }
    return "unknown";
}

ModelResolutionInfo ResolutionPolicy::defaults(models::Architecture architecture) {
    ModelResolutionInfo m;
    m.architecture = architecture;
    switch (architecture) {
        case models::Architecture::StableDiffusion15:
            m.native_width = 512; m.native_height = 512; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 64; m.min_height = 64; m.max_width = 1536; m.max_height = 1536; m.activation_multiplier = 1.0; break;
        case models::Architecture::StableDiffusion2:
            m.native_width = 768; m.native_height = 768; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 64; m.min_height = 64; m.max_width = 1536; m.max_height = 1536; m.activation_multiplier = 1.15; break;
        case models::Architecture::SDXL:
            m.native_width = 1024; m.native_height = 1024; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 128; m.min_height = 128; m.max_width = 2048; m.max_height = 2048; m.activation_multiplier = 1.45; break;
        case models::Architecture::SD3:
        case models::Architecture::SD35:
            m.native_width = 1080; m.native_height = 1080; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 256; m.min_height = 256; m.max_width = 2048; m.max_height = 2048; m.activation_multiplier = 2.20; break;
        case models::Architecture::FLUX:
        case models::Architecture::Anima:
            m.native_width = 1080; m.native_height = 1080; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 256; m.min_height = 256; m.max_width = 2048; m.max_height = 2048; m.activation_multiplier = 2.60; break;
        default:
            m.native_width = 512; m.native_height = 512; m.vae_downsample = 8; m.latent_alignment = 8;
            m.min_width = 64; m.min_height = 64; m.max_width = 1536; m.max_height = 1536; m.activation_multiplier = 1.0; break;
    }
    return m;
}

bool ResolutionPolicy::validate(const ModelResolutionInfo& model, uint32_t width, uint32_t height, std::string& error) {
    if (model.vae_downsample == 0 || model.latent_alignment == 0) { error = "invalid model VAE/alignment configuration"; return false; }
    if (width == 0 || height == 0) { error = "resolution must be non-zero"; return false; }
    if (width < model.min_width || height < model.min_height) { error = "resolution is below the model minimum"; return false; }
    if (model.max_width && width > model.max_width) { error = "width exceeds model maximum"; return false; }
    if (model.max_height && height > model.max_height) { error = "height exceeds model maximum"; return false; }
    const uint64_t factor = static_cast<uint64_t>(model.vae_downsample) * model.latent_alignment;
    if (factor == 0 || width % factor != 0 || height % factor != 0) {
        error = "resolution is not aligned to VAE latent constraints";
        return false;
    }
    return true;
}

bool ResolutionPlanner::resolve(const ModelResolutionInfo& model,
                                   const ResolutionRequest& request,
                                   ResolvedResolution& output,
                                   std::string& error) const {
    output = {};
    output.requested_width = request.width;
    output.requested_height = request.height;

    if (model.vae_downsample == 0 || model.latent_alignment == 0) {
        error = "invalid model VAE/alignment policy";
        return false;
    }
    if (request.dtype == tensor::TensorDType::Unknown || dtypeBytes(request.dtype) == 0) {
        error = "unsupported resolution planner dtype";
        return false;
    }

    const uint64_t imageAlignment64 =
        static_cast<uint64_t>(model.vae_downsample) * model.latent_alignment;
    if (imageAlignment64 == 0 || imageAlignment64 > std::numeric_limits<uint32_t>::max()) {
        error = "resolution alignment exceeds supported range";
        return false;
    }
    const uint32_t alignment = static_cast<uint32_t>(imageAlignment64);

    uint32_t requestedW = request.width ? request.width : model.native_width;
    uint32_t requestedH = request.height ? request.height : model.native_height;
    if (requestedW == 0 || requestedH == 0) {
        error = "resolution request is empty";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }
    output.requested_width = requestedW;
    output.requested_height = requestedH;
    output.reason = request.width == 0 || request.height == 0
        ? ResolutionReason::NativeDefault : ResolutionReason::UserRequested;

    if (requestedW < model.min_width || requestedH < model.min_height) {
        error = "requested resolution is below model minimum";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }
    if (model.max_width && requestedW > model.max_width) {
        error = "requested width exceeds model maximum";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }
    if (model.max_height && requestedH > model.max_height) {
        error = "requested height exceeds model maximum";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }

    // The output target is normalized to a VAE-safe image size first. This
    // lets the UI honor a request such as 1081x1081 as 1080x1080 rather than
    // silently changing it to an unrelated graph shape.
    uint32_t outputW = alignNearest(requestedW, model.vae_downsample);
    uint32_t outputH = alignNearest(requestedH, model.vae_downsample);
    if (outputW == 0 || outputH == 0) {
        error = "resolution cannot be aligned to the VAE scale";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }
    if (model.max_width && outputW > model.max_width) outputW = alignDown(model.max_width, model.vae_downsample);
    if (model.max_height && outputH > model.max_height) outputH = alignDown(model.max_height, model.vae_downsample);
    if (outputW < model.min_width || outputH < model.min_height) {
        error = "aligned output resolution is below model minimum";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }

    // UNet/DiT graph shape constraints operate on the latent. If latent
    // alignment is 8, for example, the image execution shape must be 64-wide
    // aligned. The final output may still be the VAE-aligned target above.
    uint32_t execW = alignNearest(outputW, alignment);
    uint32_t execH = alignNearest(outputH, alignment);
    if (model.max_width && execW > model.max_width) execW = alignDown(model.max_width, alignment);
    if (model.max_height && execH > model.max_height) execH = alignDown(model.max_height, alignment);
    if (execW < model.min_width || execH < model.min_height || execW == 0 || execH == 0) {
        error = "requested resolution cannot satisfy latent/UNet shape constraints";
        output.reason = ResolutionReason::InvalidRequest;
        return false;
    }

    output.output_width = outputW;
    output.output_height = outputH;
    output.width = execW;
    output.height = execH;
    output.vae_downsample = model.vae_downsample;
    output.alignment = alignment;
    output.latent_width = execW / model.vae_downsample;
    output.latent_height = execH / model.vae_downsample;

    if (output.output_width != requestedW || output.output_height != requestedH ||
        output.width != output.output_width || output.height != output.output_height) {
        output.adjusted = true;
        output.reason = ResolutionReason::AlignedToVAE;
        if (output.width != output.output_width || output.height != output.output_height) {
            output.warning = "Execution resolution aligned to model graph; final image will be resized to the requested VAE-safe size";
        } else {
            output.warning = "Resolution adjusted to the nearest VAE-safe size";
        }
    }

    bool ok = true;
    uint64_t estimate = estimatePeak(model, execW, execH, request.dtype, ok);
    if (!ok || estimate == 0) {
        error = "resolution memory estimate overflow";
        return false;
    }

    const uint64_t budget = budgetFor(request);
    output.memory_budget_bytes = budget;

    if (budget != 0 && estimate > budget) {
        const double aspect = static_cast<double>(outputW) / static_cast<double>(outputH);
        uint32_t bestW = 0, bestH = 0;
        uint64_t bestEstimate = 0;

        // Search downward in graph-aligned latent space while preserving
        // aspect ratio. No candidate is accepted unless its estimated peak
        // memory fits the selected backend budget.
        const uint32_t minH = alignNearest(model.min_height, alignment);
        for (uint32_t candidateH = alignDown(execH, alignment);
             candidateH >= minH;) {
            double candidateWf = static_cast<double>(candidateH) * aspect;
            if (candidateWf > static_cast<double>(std::numeric_limits<uint32_t>::max()))
                candidateWf = static_cast<double>(std::numeric_limits<uint32_t>::max());
            uint32_t candidateW = alignDown(static_cast<uint32_t>(candidateWf), alignment);
            if (candidateW >= model.min_width) {
                if (model.max_width) candidateW = std::min(candidateW, alignDown(model.max_width, alignment));
                bool candidateOk = true;
                const uint64_t candidateEstimate =
                    estimatePeak(model, candidateW, candidateH, request.dtype, candidateOk);
                if (candidateOk && candidateEstimate <= budget) {
                    bestW = candidateW; bestH = candidateH; bestEstimate = candidateEstimate;
                    break;
                }
            }
            if (candidateH <= minH || candidateH < alignment) break;
            candidateH -= alignment;
        }

        if (bestW == 0 || bestH == 0) {
            error = "requested resolution exceeds available memory even at the model minimum";
            output.reason = ResolutionReason::InvalidRequest;
            return false;
        }

        execW = bestW;
        execH = bestH;
        output.width = execW;
        output.height = execH;
        // A memory-driven reduction is a real output-size decision, not an
        // internal graph padding. Never claim the original target was
        // produced unless a later, explicit resize stage is actually run.
        output.output_width = execW;
        output.output_height = execH;
        output.latent_width = execW / model.vae_downsample;
        output.latent_height = execH / model.vae_downsample;
        output.adjusted = true;
        output.memory_limited = true;
        output.reason = ResolutionReason::LimitedByMemory;
        output.warning = "Resolution reduced to fit available runtime memory";
        estimate = bestEstimate;

        // The final output must never claim a size larger than the actual
        // execution frame when memory limiting occurs.
        output.output_width = std::min(output.output_width, execW);
        output.output_height = std::min(output.output_height, execH);
    }

    if (output.latent_width == 0 || output.latent_height == 0) {
        error = "resolved latent shape is empty";
        return false;
    }
    output.estimated_peak_bytes = estimate;
    return true;
}

} // namespace localimage::runtime
