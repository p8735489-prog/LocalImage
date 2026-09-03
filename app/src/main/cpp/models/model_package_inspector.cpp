#include "model_package_inspector.h"

namespace localimage::models {

bool ModelPackageInspector::inspect(const safetensors::SafeTensorFile& file, PackageInspection& output, std::string& error) const {
    output = {};
    output.tensorCount = file.tensors().size();
    if (output.tensorCount == 0) { error = "model contains no tensors"; return false; }
    for (const auto& item : file.tensors()) output.parameterBytes += item.second.byte_size;
    ModelDetector detector;
    output.detection = detector.detect(file);
    if (output.detection.architecture == Architecture::Unknown) {
        error = output.detection.reason.empty() ? "model architecture is unknown" : output.detection.reason;
        return false;
    }
    output.inventory = inventory(file, output.detection.architecture);

    // A raw .safetensors file contains weights only. It is conversion-ready when
    // the required weight families are present, but it is NOT executable until
    // tokenizer/scheduler assets and an executable graph are packaged.
    ComponentFlags detected{};
    detected.unet = !output.inventory.unet.empty();
    detected.vae = !output.inventory.vae.empty();
    detected.clip = !output.inventory.clipL.empty();
    detected.openclip = !output.inventory.openclipG.empty();
    detected.t5 = !output.inventory.t5.empty();
    detected.transformer = !output.inventory.transformer.empty();

    for (const auto& item : file.tensors()) {
        const auto& n = item.first;
        if (n.find("tokenizer") != std::string::npos) output.tokenizerEmbedded = true;
        if (n.find("scheduler") != std::string::npos) output.schedulerEmbedded = true;
    }

    output.components = validateComponents(output.detection.architecture, detected,
                                           output.tokenizerEmbedded, output.schedulerEmbedded);
    output.conversionReady = output.components.valid;
    // Raw safetensors does not contain a LocalImage executable graph. Never label
    // it executable merely because its weight families were detected.
    output.executable = false;

    if (!output.conversionReady) {
        error = output.components.error;
        return false;
    }
    output.error = "weights validated; package conversion required before execution";
    return true;
}

} // namespace localimage::models
