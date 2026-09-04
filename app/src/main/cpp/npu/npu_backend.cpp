#include "npu_backend.h"
#include "qnn/qnn_context.h"
#include "qnn/qnn_op_map.h"

#include <sstream>
#include <cstring>

#if __has_include(<android/hardware_buffer.h>)
#include <android/hardware_buffer.h>
#endif

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

namespace localimage {
namespace npu {

// ============================================================================
// Helpers
// ============================================================================

const char* BackendProbe::backendName(Backend b) {
    switch(b) {
        case Backend::None:    return "None";
        case Backend::QNN_Htp: return "QNN HTP (Hexagon DSP)";
        case Backend::QNN_Gpu: return "QNN GPU (Adreno)";
        case Backend::NNAPI:   return "Android NNAPI";
        default: return "Unknown";
    }
}

const char* BackendProbe::dspVersionName(DspVersion v) {
    switch(v) {
        case DspVersion::V73: return "v73 (Snapdragon 8 Gen2)";
        case DspVersion::V75: return "v75 (Snapdragon 8 Gen3)";
        case DspVersion::V79: return "v79 (Snapdragon 8 Elite)";
        case DspVersion::V81: return "v81 (Snapdragon 8 Elite Gen5)";
        default: return "unknown";
    }
}

bool BackendProbe::meetsMinimumRequirement(DspVersion v) {
    // Minimum: 8 Gen3 = Hexagon v75
    return static_cast<int>(v) >= static_cast<int>(DspVersion::V75);
}

// ============================================================================
// SoC detection
// ============================================================================

static std::string readSystemProperty(const char* key) {
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = {0};
    int len = __system_property_get(key, value);
    if (len > 0) return std::string(value, len);
#else
    (void)key;
#endif
    return "";
}

std::string BackendProbe::detectSocModel() {
    // Try multiple system properties in order of reliability
    std::string soc = readSystemProperty("ro.soc.model");
    if (!soc.empty()) return soc;

    soc = readSystemProperty("ro.board.platform");
    if (!soc.empty()) return soc;

    soc = readSystemProperty("ro.product.board");
    if (!soc.empty()) return soc;

    soc = readSystemProperty("ro.hardware");
    if (!soc.empty()) return soc;

    soc = readSystemProperty("ro.product.device");
    if (!soc.empty()) return soc;

    return "unknown";
}

DspVersion BackendProbe::socToDspVersion(const std::string& socModel) {
    // Qualcomm Snapdragon SoC -> Hexagon DSP version mapping
    //
    // Snapdragon 8 Elite Gen5  -> SM8850 -> Hexagon v81
    // Snapdragon 8 Elite       -> SM8750 -> Hexagon v79
    // Snapdragon 8 Gen3        -> SM8650 -> Hexagon v75
    // Snapdragon 8 Gen2        -> SM8550 -> Hexagon v73

    if (socModel.find("SM8850") != std::string::npos) {
        return DspVersion::V81;
    }
    if (socModel.find("8 Elite Gen5") != std::string::npos ||
        socModel.find("8 Elite 2") != std::string::npos) {
        return DspVersion::V81;
    }
    if (socModel.find("SM8750") != std::string::npos) {
        return DspVersion::V79;
    }
    if (socModel.find("8 Elite") != std::string::npos) {
        return DspVersion::V79;
    }
    if (socModel.find("SM8650") != std::string::npos) {
        return DspVersion::V75;
    }
    if (socModel.find("8 Gen3") != std::string::npos ||
        socModel.find("8gen3") != std::string::npos ||
        socModel.find("8 Gen 3") != std::string::npos) {
        return DspVersion::V75;
    }
    if (socModel.find("SM8550") != std::string::npos) {
        return DspVersion::V73;
    }
    if (socModel.find("8 Gen2") != std::string::npos ||
        socModel.find("8 Gen 2") != std::string::npos) {
        return DspVersion::V73;
    }
    return DspVersion::Unknown;
}

// ============================================================================
// QNN HTP probe
// ============================================================================

bool BackendProbe::probeQnnHtp(Capabilities& caps) {
#ifndef LOCALIMAGE_NO_NPU
    std::string error;
    auto& ctx = qnn::getQnnContext();

    // Initialize QNN context (this loads the backend and creates device)
    if (!ctx.initialize(error)) {
        caps.errorMessage = error;
        return false;
    }

    const auto& qnnCaps = ctx.capabilities();
    caps.available = qnnCaps.available;
    caps.backend = Backend::QNN_Htp;
    caps.dspVersion = qnnCaps.dspVersion;
    caps.device = qnnCaps.deviceName;
    caps.totalMemoryBytes = qnnCaps.totalNpuMemoryBytes;

    // Get supported ops from QNN
    qnn::OpMapper mapper;
    caps.supportedOps = mapper.supportedOpNames();

    return caps.available;
#else
    caps.errorMessage = "QNN NPU backend not compiled in (LOCALIMAGE_NO_NPU defined)";
    return false;
#endif
}

// ============================================================================
// Main detect entry point
// ============================================================================

Capabilities BackendProbe::detect() {
    Capabilities caps;

    // Step 1: Detect SoC model
    caps.socName = detectSocModel();
    caps.dspVersion = socToDspVersion(caps.socName);

    // Step 2: Check minimum DSP version requirement
    if (caps.dspVersion == DspVersion::Unknown) {
        caps.available = false;
        caps.backend = Backend::None;
        caps.device = "NPU capability probe: cannot identify Qualcomm SoC. SoC: " + caps.socName;
        caps.errorMessage = "Unknown SoC model: " + caps.socName;
        return caps;
    }

    if (!meetsMinimumRequirement(caps.dspVersion)) {
        caps.available = false;
        caps.backend = Backend::None;
        std::ostringstream os;
        os << "NPU capability probe: DSP version "
           << dspVersionName(caps.dspVersion)
           << " is below minimum required "
           << dspVersionName(DspVersion::V75)
           << " (Snapdragon 8 Gen3)";
        caps.device = os.str();
        caps.errorMessage = os.str();
        return caps;
    }

    // Step 3: Probe QNN HTP backend (primary)
    if (probeQnnHtp(caps)) {
        return caps;
    }

    // Step 4: QNN HTP not available, record why
    caps.available = false;
    caps.backend = Backend::None;
    if (caps.device.empty()) {
        std::ostringstream os;
        os << "NPU capability probe: QNN HTP backend unavailable. "
           << "SoC: " << caps.socName
           << ", DSP: " << dspVersionName(caps.dspVersion);
        if (!caps.errorMessage.empty()) {
            os << ". Reason: " << caps.errorMessage;
        }
        caps.device = os.str();
    }

    return caps;
}

} // namespace npu
} // namespace localimage
