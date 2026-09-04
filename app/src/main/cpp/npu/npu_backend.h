#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace localimage {
namespace npu {

// NPU backend type
enum class Backend {
    None = 0,
    QNN_Htp = 1,   // Qualcomm QNN HTP (Hexagon DSP) - 首选
    QNN_Gpu = 2,   // Qualcomm QNN GPU (Adreno) - 备选
    NNAPI   = 3,   // Android NNAPI - 通用
};

// Hexagon DSP version
enum class DspVersion {
    Unknown = 0,
    V73 = 73,   // Snapdragon 8 Gen2
    V75 = 75,   // Snapdragon 8 Gen3  (最低支持版本)
    V79 = 79,   // Snapdragon 8 Elite
    V81 = 81,   // Snapdragon 8 Elite Gen5
};

// NPU capability probe result
struct Capabilities {
    bool available = false;
    Backend backend = Backend::None;
    DspVersion dspVersion = DspVersion::Unknown;
    std::string socName;       // SoC 型号 (如 "SM8650")
    std::string device;        // 设备描述 (如 "Qualcomm Hexagon DSP v75 (8 Gen3)")
    std::vector<std::string> supportedOps; // 支持的算子列表
    uint64_t totalMemoryBytes = 0; // 可用 NPU 内存
    std::string errorMessage;  // 不可用时的错误原因
};

// Backend probe: detects available NPU hardware and capabilities
class BackendProbe {
public:
    // Detect and return NPU capabilities
    // Always returns a Capabilities struct (available=false if none found)
    static Capabilities detect();

    // Get backend name as string
    static const char* backendName(Backend b);

    // Get DSP version name
    static const char* dspVersionName(DspVersion v);

    // Check if DSP version meets minimum requirement (8 Gen3 = v75)
    static bool meetsMinimumRequirement(DspVersion v);

private:
    // Detect SoC model from system properties
    static std::string detectSocModel();

    // Map SoC model to DSP version
    static DspVersion socToDspVersion(const std::string& socModel);

    // Probe QNN HTP backend availability
    static bool probeQnnHtp(Capabilities& caps);
};

} // namespace npu
} // namespace localimage
