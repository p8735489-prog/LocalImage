#include "qnn_types.h"

namespace localimage {
namespace npu {
namespace qnn {

// ============================================================================
// Dtype conversion
// ============================================================================

QnnDataType fromTensorDtype(int tensorDtype) {
    // 0=F32, 1=F16, 2=BF16 per existing tensor convention
    switch(tensorDtype) {
        case 0: return QnnDataType::Float32;
        case 1: return QnnDataType::Float16;
        case 2: return QnnDataType::BFloat16;
        default: return QnnDataType::Float16;
    }
}

int toTensorDtype(QnnDataType qnnDtype) {
    switch(qnnDtype) {
        case QnnDataType::Float32:  return 0;
        case QnnDataType::Float16:  return 1;
        case QnnDataType::BFloat16: return 2;
        case QnnDataType::Int8:     return 3;
        default: return 1; // default F16
    }
}

// ============================================================================
// DSP version
// ============================================================================

std::string dspVersionName(DspVersion v) {
    switch(v) {
        case DspVersion::V73: return "v73 (8 Gen2)";
        case DspVersion::V75: return "v75 (8 Gen3)";
        case DspVersion::V79: return "v79 (8 Elite)";
        case DspVersion::V81: return "v81 (8 Elite Gen5)";
        default: return "unknown";
    }
}

} // namespace qnn
} // namespace npu
} // namespace localimage
