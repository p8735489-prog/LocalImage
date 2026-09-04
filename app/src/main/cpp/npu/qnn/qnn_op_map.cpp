#include "qnn_op_map.h"
#include <algorithm>

namespace localimage {
namespace npu {
namespace qnn {

// ============================================================================
// OpMapper
// ============================================================================

OpMapper::OpMapper() {
    buildMappings();
}

OpMapper::~OpMapper() = default;

void OpMapper::buildMappings() {
    // Direct 1:1 mappings (LocalImage IR Op -> QNN Op)
    directMap_[ir::Op::Add]             = QnnOpType::Add;
    directMap_[ir::Op::Sub]             = QnnOpType::Sub;
    directMap_[ir::Op::Mul]             = QnnOpType::Mul;
    directMap_[ir::Op::Div]             = QnnOpType::Div;
    directMap_[ir::Op::MatMul]          = QnnOpType::MatMul;
    directMap_[ir::Op::BatchedMatMul]   = QnnOpType::BatchMatMul;
    directMap_[ir::Op::Conv2D]          = QnnOpType::Conv2D;
    directMap_[ir::Op::Softmax]         = QnnOpType::Softmax;
    directMap_[ir::Op::LayerNorm]       = QnnOpType::LayerNorm;
    directMap_[ir::Op::RMSNorm]         = QnnOpType::RMSNorm;
    directMap_[ir::Op::GroupNorm]       = QnnOpType::GroupNorm;
    directMap_[ir::Op::SiLU]            = QnnOpType::SiLU;
    directMap_[ir::Op::GELU]            = QnnOpType::GELU;
    directMap_[ir::Op::Exp]             = QnnOpType::Exp;
    directMap_[ir::Op::Sqrt]            = QnnOpType::Sqrt;
    directMap_[ir::Op::Rsqrt]           = QnnOpType::Rsqrt;
    directMap_[ir::Op::Clamp]           = QnnOpType::Clamp;
    directMap_[ir::Op::Reshape]         = QnnOpType::Reshape;
    directMap_[ir::Op::Transpose]       = QnnOpType::Transpose;
    directMap_[ir::Op::Slice]           = QnnOpType::Slice;
    directMap_[ir::Op::Concat]          = QnnOpType::Concat;
    directMap_[ir::Op::Broadcast]       = QnnOpType::Broadcast;
    directMap_[ir::Op::Upsample]        = QnnOpType::UpsampleNearest;
    directMap_[ir::Op::RoPE]            = QnnOpType::RoPE;
    directMap_[ir::Op::Attention]       = QnnOpType::Attention;
    // Linear is handled as decomposed (MatMul + Add)
}

OpMappingResult OpMapper::mapOp(ir::Op op, const ir::Attributes& attr) const {
    OpMappingResult result;

    // Input is not a compute op
    if (op == ir::Op::Input) {
        result.supported = false;
        result.error = "Input is not a compute operation";
        return result;
    }

    // Check direct mapping first
    auto it = directMap_.find(op);
    if (it != directMap_.end()) {
        result.supported = true;
        result.targetOp = it->second;
        return result;
    }

    // Handle special cases / decomposed ops
    if (op == ir::Op::Linear) {
        // Linear = MatMul + Add (bias)
        result.supported = true;
        result.targetOp = QnnOpType::MatMul;
        result.decomposed = true;
        result.decomposedOps = { QnnOpType::MatMul, QnnOpType::Add };
        return result;
    }

    result.supported = false;
    result.error = "Unsupported op for QNN backend";
    return result;
}

bool OpMapper::isSupported(ir::Op op) const {
    if (op == ir::Op::Input) return false;
    if (op == ir::Op::Linear) return true; // decomposed

    auto it = directMap_.find(op);
    return it != directMap_.end();
}

std::vector<std::string> OpMapper::supportedOpNames() const {
    std::vector<std::string> names;
    for (const auto& [op, qnnOp] : directMap_) {
        names.push_back(ir::opName(op));
    }
    names.push_back("Linear"); // decomposed
    std::sort(names.begin(), names.end());
    return names;
}

const char* OpMapper::qnnOpName(QnnOpType op) {
    switch(op) {
        case QnnOpType::Input:           return "Input";
        case QnnOpType::Output:          return "Output";
        case QnnOpType::Add:             return "Add";
        case QnnOpType::Sub:             return "Sub";
        case QnnOpType::Mul:             return "Mul";
        case QnnOpType::Div:             return "Div";
        case QnnOpType::MatMul:          return "MatMul";
        case QnnOpType::BatchMatMul:     return "BatchMatMul";
        case QnnOpType::Conv2D:          return "Conv2D";
        case QnnOpType::DepthwiseConv2D: return "DepthwiseConv2D";
        case QnnOpType::TransposeConv2D: return "TransposeConv2D";
        case QnnOpType::Softmax:         return "Softmax";
        case QnnOpType::LayerNorm:       return "LayerNorm";
        case QnnOpType::RMSNorm:         return "RMSNorm";
        case QnnOpType::GroupNorm:       return "GroupNorm";
        case QnnOpType::SiLU:            return "SiLU";
        case QnnOpType::GELU:            return "GELU";
        case QnnOpType::ReLU:            return "ReLU";
        case QnnOpType::LeakyReLU:       return "LeakyReLU";
        case QnnOpType::Exp:             return "Exp";
        case QnnOpType::Sqrt:            return "Sqrt";
        case QnnOpType::Rsqrt:           return "Rsqrt";
        case QnnOpType::Clamp:           return "Clamp";
        case QnnOpType::Reshape:         return "Reshape";
        case QnnOpType::Transpose:       return "Transpose";
        case QnnOpType::Slice:           return "Slice";
        case QnnOpType::Concat:          return "Concat";
        case QnnOpType::Broadcast:       return "Broadcast";
        case QnnOpType::UpsampleNearest: return "UpsampleNearest";
        case QnnOpType::UpsampleBilinear:return "UpsampleBilinear";
        case QnnOpType::RoPE:            return "RoPE";
        case QnnOpType::Attention:       return "Attention";
        case QnnOpType::ReduceMean:      return "ReduceMean";
        case QnnOpType::ReduceSum:       return "ReduceSum";
        case QnnOpType::Sigmoid:         return "Sigmoid";
        case QnnOpType::Tanh:            return "Tanh";
        case QnnOpType::Pad:             return "Pad";
        case QnnOpType::StridedSlice:    return "StridedSlice";
        case QnnOpType::Gather:          return "Gather";
        case QnnOpType::Squeeze:         return "Squeeze";
        case QnnOpType::Unsqueeze:       return "Unsqueeze";
        default: return "Unknown";
    }
}

// ============================================================================
// OpConfigBuilder
// ============================================================================

OpConfigBuilder::OpConfigBuilder() = default;
OpConfigBuilder::~OpConfigBuilder() = default;

OpConfigBuilder::MatMulConfig OpConfigBuilder::buildMatMul(const ir::Attributes& /*attr*/) {
    MatMulConfig cfg;
    // LocalImage MatMul is standard (no transpose flags in IR currently)
    return cfg;
}

OpConfigBuilder::Conv2DConfig OpConfigBuilder::buildConv2D(const ir::Attributes& attr) {
    Conv2DConfig cfg;
    cfg.strideH = static_cast<uint32_t>(attr.stride);
    cfg.strideW = static_cast<uint32_t>(attr.stride);
    cfg.padH = static_cast<uint32_t>(attr.padding);
    cfg.padW = static_cast<uint32_t>(attr.padding);
    cfg.dilationH = static_cast<uint32_t>(attr.dilation);
    cfg.dilationW = static_cast<uint32_t>(attr.dilation);
    cfg.groups = static_cast<uint32_t>(attr.groups);
    return cfg;
}

OpConfigBuilder::SoftmaxConfig OpConfigBuilder::buildSoftmax(const ir::Attributes& attr) {
    SoftmaxConfig cfg;
    if (!attr.axes.empty()) {
        cfg.axis = static_cast<int32_t>(attr.axes[0]);
    }
    return cfg;
}

OpConfigBuilder::NormConfig OpConfigBuilder::buildLayerNorm(const ir::Attributes& attr) {
    NormConfig cfg;
    cfg.epsilon = static_cast<float>(attr.epsilon);
    return cfg;
}

OpConfigBuilder::NormConfig OpConfigBuilder::buildRMSNorm(const ir::Attributes& attr) {
    NormConfig cfg;
    cfg.epsilon = static_cast<float>(attr.epsilon);
    return cfg;
}

OpConfigBuilder::NormConfig OpConfigBuilder::buildGroupNorm(const ir::Attributes& attr) {
    NormConfig cfg;
    cfg.epsilon = static_cast<float>(attr.epsilon);
    cfg.groups = static_cast<uint32_t>(attr.groups);
    return cfg;
}

OpConfigBuilder::ResizeConfig OpConfigBuilder::buildUpsample(const ir::Attributes& attr) {
    ResizeConfig cfg;
    cfg.scaleFactor = static_cast<uint32_t>(attr.scale_factor);
    // LocalImage Upsample is nearest-neighbor by default
    cfg.mode = "nearest";
    return cfg;
}

OpConfigBuilder::SliceConfig OpConfigBuilder::buildSlice(const ir::Attributes& attr,
                                                          uint32_t rank) {
    SliceConfig cfg;
    cfg.starts.reserve(attr.slice_starts.size());
    cfg.lengths.reserve(attr.slice_lengths.size());
    for (auto s : attr.slice_starts) {
        cfg.starts.push_back(static_cast<uint32_t>(s));
    }
    for (auto l : attr.slice_lengths) {
        cfg.lengths.push_back(static_cast<uint32_t>(l));
    }
    // Pad if needed
    while (cfg.starts.size() < rank) cfg.starts.push_back(0);
    while (cfg.lengths.size() < rank) cfg.lengths.push_back(0);
    return cfg;
}

OpConfigBuilder::ConcatConfig OpConfigBuilder::buildConcat(const ir::Attributes& attr) {
    ConcatConfig cfg;
    if (!attr.axes.empty()) {
        cfg.axis = static_cast<uint32_t>(attr.axes[0]);
    }
    return cfg;
}

OpConfigBuilder::TransposeConfig OpConfigBuilder::buildTranspose(const ir::Attributes& attr) {
    TransposeConfig cfg;
    cfg.perm.reserve(attr.permutation.size());
    for (auto p : attr.permutation) {
        cfg.perm.push_back(static_cast<uint32_t>(p));
    }
    return cfg;
}

OpConfigBuilder::ClampConfig OpConfigBuilder::buildClamp() {
    ClampConfig cfg;
    // Default values - actual min/max would come from op attributes
    cfg.minValue = 0.0f;
    cfg.maxValue = 1.0f;
    return cfg;
}

OpConfigBuilder::AttentionConfig OpConfigBuilder::buildAttention(const ir::Attributes& attr) {
    AttentionConfig cfg;
    cfg.scale = static_cast<float>(attr.attention_scale);
    cfg.hasMask = false; // determined by input count at runtime
    return cfg;
}

} // namespace qnn
} // namespace npu
} // namespace localimage
