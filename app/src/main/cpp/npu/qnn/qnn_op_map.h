#pragma once

#include "qnn_types.h"
#include "../../runtime/ir/localimage_ir.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace localimage {
namespace npu {
namespace qnn {

// Op mapping result
struct OpMappingResult {
    bool supported = false;
    QnnOpType targetOp;
    std::string error;
    // For ops that decompose into multiple QNN ops:
    bool decomposed = false;
    std::vector<QnnOpType> decomposedOps;
};

// OpMapper: bridges LocalImage IR Op -> QNN Op
// This is the "转接头" (adapter) core logic.
// Key design:
// - Direct 1:1 mapping when possible (MatMul -> MatMul, Conv2D -> Conv2D)
// - Decomposition when needed (e.g., Linear -> MatMul + Add)
// - Reports unsupported ops so the graph can fall back to CPU/Vulkan
class OpMapper {
public:
    OpMapper();
    ~OpMapper();

    // Map a LocalImage IR op to QNN op(s)
    OpMappingResult mapOp(ir::Op op, const ir::Attributes& attr) const;

    // Check if an op is supported by QNN HTP backend
    bool isSupported(ir::Op op) const;

    // Get list of all supported ops (for capability reporting)
    std::vector<std::string> supportedOpNames() const;

    // Get QNN op name string
    static const char* qnnOpName(QnnOpType op);

private:
    // Direct mapping table (1:1)
    std::unordered_map<ir::Op, QnnOpType> directMap_;

    // Build the mapping tables
    void buildMappings();
};

// Op config builder: translates LocalImage IR attributes to QNN op parameters
// Each op type has its own config struct in QNN SDK
class OpConfigBuilder {
public:
    OpConfigBuilder();
    ~OpConfigBuilder();

    // Elementwise ops (Add, Sub, Mul, Div)
    // No special params needed - QNN infers from inputs

    // MatMul params
    struct MatMulConfig {
        bool transposeA = false;
        bool transposeB = false;
    };

    static MatMulConfig buildMatMul(const ir::Attributes& attr);

    // Conv2D params
    struct Conv2DConfig {
        uint32_t strideH = 1;
        uint32_t strideW = 1;
        uint32_t padH = 0;
        uint32_t padW = 0;
        uint32_t dilationH = 1;
        uint32_t dilationW = 1;
        uint32_t groups = 1;
    };

    static Conv2DConfig buildConv2D(const ir::Attributes& attr);

    // Softmax params
    struct SoftmaxConfig {
        int32_t axis = -1;
    };

    static SoftmaxConfig buildSoftmax(const ir::Attributes& attr);

    // Norm params (LayerNorm, RMSNorm, GroupNorm)
    struct NormConfig {
        float epsilon = 1e-5f;
        uint32_t groups = 32;  // for GroupNorm
        int32_t axis = -1;
    };

    static NormConfig buildLayerNorm(const ir::Attributes& attr);
    static NormConfig buildRMSNorm(const ir::Attributes& attr);
    static NormConfig buildGroupNorm(const ir::Attributes& attr);

    // Pool / Upsample params
    struct ResizeConfig {
        uint32_t scaleFactor = 2;
        std::string mode = "nearest"; // nearest, bilinear
    };

    static ResizeConfig buildUpsample(const ir::Attributes& attr);

    // Slice params
    struct SliceConfig {
        std::vector<uint32_t> starts;
        std::vector<uint32_t> lengths;
    };

    static SliceConfig buildSlice(const ir::Attributes& attr,
                                   uint32_t rank);

    // Concat params
    struct ConcatConfig {
        uint32_t axis = 0;
    };

    static ConcatConfig buildConcat(const ir::Attributes& attr);

    // Transpose params
    struct TransposeConfig {
        std::vector<uint32_t> perm;
    };

    static TransposeConfig buildTranspose(const ir::Attributes& attr);

    // Clamp params
    struct ClampConfig {
        float minValue = 0.0f;
        float maxValue = 0.0f;
    };

    static ClampConfig buildClamp();

    // Attention params (composite op)
    struct AttentionConfig {
        float scale = 0.0f;
        bool hasMask = false;
    };

    static AttentionConfig buildAttention(const ir::Attributes& attr);

    // RoPE params
    struct RoPEConfig {
        // RoPE is decomposed into: slice + sin + cos + mul + add
    };
};

} // namespace qnn
} // namespace npu
} // namespace localimage
