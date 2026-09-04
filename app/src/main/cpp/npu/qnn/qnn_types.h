#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace localimage {
namespace npu {
namespace qnn {

// Hexagon DSP version -> Qualcomm SoC mapping
enum class DspVersion {
    Unknown = 0,
    V73 = 73,   // Snapdragon 8 Gen2
    V75 = 75,   // Snapdragon 8 Gen3  (最低目标)
    V79 = 79,   // Snapdragon 8 Elite
    V81 = 81,   // Snapdragon 8 Elite Gen5
};

// NPU backend type
enum class BackendType {
    HTP = 0,    // Hexagon Tensor Processor (DSP-based, recommended)
    GPU = 1,    // Adreno GPU (fallback)
    CPU = 2,    // CPU reference (not used - we have our own CPU backend)
};

// Tensor data type mapping
enum class QnnDataType {
    Float32 = 0,
    Float16 = 1,
    Int32   = 2,
    Int16   = 3,
    Int8    = 4,
    UInt8   = 5,
    Bool    = 6,
    BFloat16 = 7,
};

// Tensor layout
enum class QnnLayout {
    NCHW = 0,
    NHWC = 1,
    NCHW_C8 = 2,   // Block format for HTP efficiency
    NCHW_C16 = 3,
    NCHW_VEC8 = 4,
    NCHW_VEC16 = 5,
};

// Supported QNN operations
enum class QnnOpType {
    Input = 0,
    Output,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    BatchMatMul,
    Conv2D,
    DepthwiseConv2D,
    TransposeConv2D,
    Softmax,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    SiLU,
    GELU,
    ReLU,
    LeakyReLU,
    Exp,
    Sqrt,
    Rsqrt,
    Clamp,
    Reshape,
    Transpose,
    Slice,
    Concat,
    Broadcast,
    UpsampleNearest,
    UpsampleBilinear,
    RoPE,
    Attention,
    ReduceMean,
    ReduceSum,
    Sigmoid,
    Tanh,
    Pad,
    StridedSlice,
    Gather,
    Squeeze,
    Unsqueeze,
};

struct QnnTensorInfo {
    std::string name;
    std::vector<uint32_t> dims;
    QnnDataType dtype = QnnDataType::Float16;
    QnnLayout layout = QnnLayout::NCHW;
    void* dataPtr = nullptr;
    size_t dataSize = 0;
    bool isConst = false;
};

// QNN capability probe result
struct QnnCapabilities {
    bool available = false;
    BackendType backend = BackendType::HTP;
    DspVersion dspVersion = DspVersion::Unknown;
    std::string socName;
    std::string deviceName;
    uint64_t totalNpuMemoryBytes = 0;
    std::vector<QnnOpType> supportedOps;
    std::string errorMessage;
};

// Convert between LocalImage dtype and QNN dtype
QnnDataType fromTensorDtype(int tensorDtype);
int toTensorDtype(QnnDataType qnnDtype);

// Get DSP version string
std::string dspVersionName(DspVersion v);

// Get minimum supported DSP version (8 Gen3 = v75)
constexpr DspVersion kMinSupportedDspVersion = DspVersion::V75;

} // namespace qnn
} // namespace npu
} // namespace localimage
