#include "qnn_graph_executor.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <functional>

#ifdef __ANDROID__
#include <dlfcn.h>
#endif

namespace localimage {
namespace npu {
namespace qnn {

// ============================================================================
// QNN interface accessor (shared between all executors)
// ============================================================================
// Since QnnGraphExecutor cannot directly access QnnContext's private
// qnnInterface_ (and we cannot modify headers), we retrieve the QNN
// interface via dlsym from the already-loaded backend library.
// This is a standard pattern in plugin-based architectures: the backend
// library is loaded once by QnnContext, and subsequent components resolve
// the interface symbol from the global symbol table.

#ifdef LOCALIMAGE_QNN

namespace {

// Cache the QNN interface pointer after first lookup
static QnnInterface_t* g_cachedQnnInterface = nullptr;

QnnInterface_t* getQnnInterface(std::string& error) {
    if (g_cachedQnnInterface) {
        return g_cachedQnnInterface;
    }

    // Resolve QnnInterface_getProviders from the already-loaded backend
    using GetQnnInterface_fn = Qnn_ErrorHandle_t (*)(
        uint32_t, const QnnInterface_t**, uint32_t*);

    auto getInterface = reinterpret_cast<GetQnnInterface_fn>(
        dlsym(RTLD_DEFAULT, "QnnInterface_getProviders"));
    if (!getInterface) {
        error = "QNN backend not loaded: cannot resolve QnnInterface_getProviders: " +
                std::string(dlerror());
        return nullptr;
    }

    const QnnInterface_t* providers = nullptr;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t status = getInterface(
        QNN_API_VERSION, &providers, &numProviders);
    if (status != QNN_SUCCESS || !providers || numProviders == 0) {
        error = "Failed to obtain QNN interface from loaded backend";
        return nullptr;
    }

    g_cachedQnnInterface = const_cast<QnnInterface_t*>(providers);
    return g_cachedQnnInterface;
}

// Helper: build a tensor shape signature string for cache key generation
// Encodes rank, dimensions, and data type into a compact string
std::string tensorSignature(const QnnTensorInfo& info) {
    std::ostringstream os;
    os << static_cast<int>(info.dtype) << "x" << info.dims.size();
    for (auto d : info.dims) {
        os << "_" << d;
    }
    return os.str();
}

std::string graphCacheKey(const std::string& opName,
                          const std::vector<QnnTensorInfo>& inputs,
                          const QnnTensorInfo& output) {
    std::ostringstream os;
    os << opName << "|";
    for (const auto& in : inputs) {
        os << tensorSignature(in) << ",";
    }
    os << "|" << tensorSignature(output);
    return os.str();
}

// Helper: map our QnnDataType to QNN SDK QnnDataType_t
QnnDataType_t toQnnDataType(QnnDataType dtype) {
    switch(dtype) {
        case QnnDataType::Float32:  return QNN_DATATYPE_FLOAT_32;
        case QnnDataType::Float16:  return QNN_DATATYPE_FLOAT_16;
        case QnnDataType::BFloat16: return QNN_DATATYPE_BFLOAT_16;
        case QnnDataType::Int32:    return QNN_DATATYPE_INT_32;
        case QnnDataType::Int16:    return QNN_DATATYPE_INT_16;
        case QnnDataType::Int8:     return QNN_DATATYPE_INT_8;
        case QnnDataType::UInt8:    return QNN_DATATYPE_UINT_8;
        case QnnDataType::Bool:     return QNN_DATATYPE_BOOL_8;
        default: return QNN_DATATYPE_FLOAT_16;
    }
}

// Helper: fill a Qnn_Tensor_t descriptor from our QnnTensorInfo
// for graph building (defines tensor properties, no data pointer for non-const)
void fillQnnTensorDesc(Qnn_Tensor_t& tensor,
                       const QnnTensorInfo& info,
                       QnnTensorType_t tensorType) {
    tensor.name = info.name.c_str();
    tensor.type = tensorType;
    tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_NCHW; // default, adjusted per layout
    tensor.dataType = toQnnDataType(info.dtype);
    tensor.rank = static_cast<uint32_t>(info.dims.size());
    // dimensions will point to our dims data; caller must ensure lifetime
    // We use const_cast since QNN SDK may expect non-const in some versions
    tensor.dimensions = const_cast<uint32_t*>(info.dims.data());

    // For const tensors (weights/biases), set the data buffer
    if (info.isConst && info.dataPtr) {
        tensor.clientBuf.data = info.dataPtr;
        tensor.clientBuf.dataSize = info.dataSize;
        tensor.clientBuf.memType = QNN_MEM_TYPE_DEFAULT;
    } else {
        tensor.clientBuf.data = nullptr;
        tensor.clientBuf.dataSize = 0;
        tensor.clientBuf.memType = QNN_MEM_TYPE_DEFAULT;
    }

    // Layout-specific data format
    if (info.layout == QnnLayout::NHWC) {
        tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_NHWC;
    }
    // Block formats (NCHW_C8, NCHW_C16) would use vendor-specific format enums
}

// Helper: fill a Qnn_Tensor_t for execution (with data pointer set)
void fillQnnTensorExec(Qnn_Tensor_t& tensor,
                       const QnnTensorInfo& info,
                       QnnTensorType_t tensorType) {
    fillQnnTensorDesc(tensor, info, tensorType);
    // Set data pointer for execution
    tensor.clientBuf.data = info.dataPtr;
    tensor.clientBuf.dataSize = info.dataSize;
    tensor.clientBuf.memType = QNN_MEM_TYPE_DEFAULT;
}

// Map QnnOpType to QNN SDK op type name string
const char* toQnnOpTypeName(QnnOpType op) {
    switch(op) {
        case QnnOpType::Add:             return "ElementWiseAdd";
        case QnnOpType::Sub:             return "ElementWiseSubtract";
        case QnnOpType::Mul:             return "ElementWiseMultiply";
        case QnnOpType::Div:             return "ElementWiseDivide";
        case QnnOpType::MatMul:          return "MatMul";
        case QnnOpType::BatchMatMul:     return "BatchMatMul";
        case QnnOpType::Conv2D:          return "Conv2d";
        case QnnOpType::DepthwiseConv2D: return "DepthwiseConv2d";
        case QnnOpType::TransposeConv2D: return "TransposeConv2d";
        case QnnOpType::Softmax:         return "Softmax";
        case QnnOpType::LayerNorm:       return "LayerNorm";
        case QnnOpType::RMSNorm:         return "RMSNorm";
        case QnnOpType::GroupNorm:       return "GroupNorm";
        case QnnOpType::SiLU:            return "SiLU";
        case QnnOpType::GELU:            return "GELU";
        case QnnOpType::ReLU:            return "Relu";
        case QnnOpType::LeakyReLU:       return "LeakyRelu";
        case QnnOpType::Exp:             return "Exp";
        case QnnOpType::Sqrt:            return "Sqrt";
        case QnnOpType::Rsqrt:           return "Rsqrt";
        case QnnOpType::Clamp:           return "Clamp";
        case QnnOpType::Reshape:         return "Reshape";
        case QnnOpType::Transpose:       return "Transpose";
        case QnnOpType::Slice:           return "Slice";
        case QnnOpType::Concat:          return "Concat";
        case QnnOpType::Broadcast:       return "Broadcast";
        case QnnOpType::UpsampleNearest: return "ResizeNearestNeighbor";
        case QnnOpType::UpsampleBilinear:return "ResizeBilinear";
        case QnnOpType::ReduceMean:      return "ReduceMean";
        case QnnOpType::ReduceSum:       return "ReduceSum";
        case QnnOpType::Sigmoid:         return "Sigmoid";
        case QnnOpType::Tanh:            return "Tanh";
        case QnnOpType::Pad:             return "Pad";
        case QnnOpType::StridedSlice:    return "StridedSlice";
        case QnnOpType::Gather:          return "Gather";
        case QnnOpType::Squeeze:         return "Squeeze";
        case QnnOpType::Unsqueeze:       return "Unsqueeze";
        default: return nullptr;
    }
}

} // anonymous namespace
#endif // LOCALIMAGE_QNN

// ============================================================================
// QnnGraphExecutor
// ============================================================================

QnnGraphExecutor::QnnGraphExecutor() = default;

QnnGraphExecutor::~QnnGraphExecutor() {
    clear();
}

bool QnnGraphExecutor::initialize(QnnContext& context, std::string& error) {
    if (ready_) {
        error = "QNN graph executor already initialized";
        return false;
    }

    if (!context.isAvailable()) {
        error = "QNN context is not available";
        return false;
    }

    context_ = &context;
    ready_ = true;
    return true;
}

void QnnGraphExecutor::clear() {
#ifdef LOCALIMAGE_QNN
    // Free all cached QNN graph handles properly using the QNN interface
    std::string ignoredError;
    QnnInterface_t* qnnIface = getQnnInterface(ignoredError);
    if (qnnIface && qnnIface->graph) {
        for (auto& [key, cache] : opCache_) {
            if (cache.graphHandle) {
                qnnIface->graph->free(cache.graphHandle);
                cache.graphHandle = nullptr;
            }
        }
    } else {
        // If interface is unavailable (e.g., already unloaded), just null out
        // the handles to avoid dangling pointers
        for (auto& [key, cache] : opCache_) {
            cache.graphHandle = nullptr;
        }
    }
#endif
    opCache_.clear();
    ready_ = false;
    context_ = nullptr;
}

// ============================================================================
// Single-op execution (main entry point)
// ============================================================================

bool QnnGraphExecutor::executeSingleOp(ir::Op op,
                                        const ir::Attributes& attr,
                                        const std::vector<tensor::Tensor>& inputs,
                                        tensor::Tensor& output,
                                        std::string& error) {
    if (!ready_) {
        error = "QNN graph executor not initialized";
        return false;
    }

    // Check if op is supported
    OpMappingResult mapping = opMapper_.mapOp(op, attr);
    if (!mapping.supported) {
        error = "QNN backend does not support op: " + std::string(ir::opName(op));
        return false;
    }

    // Infer output shape
    tensor::TensorShape outShape;
    tensor::TensorDType outDtype = tensor::TensorDType::F16;
    if (!inferOutputShape(op, attr, inputs, outShape, outDtype, error)) {
        return false;
    }

    // Special case: Reshape is a metadata-only op, no NPU needed
    if (op == ir::Op::Reshape) {
        if (inputs.size() != 1) {
            error = "Reshape requires exactly one input";
            return false;
        }
        tensor::TensorRuntime rt;
        return rt.reshape(inputs[0], outShape, output, error);
    }

    // Prepare QNN input tensor infos
    std::vector<QnnTensorInfo> qnnInputs;
    std::vector<void*> tempAllocs;
    if (!prepareQnnInputs(inputs, qnnInputs, tempAllocs, error)) {
        cleanupTempAllocations(tempAllocs);
        return false;
    }

    // Build output tensor info
    QnnTensorInfo qnnOutput;
    qnnOutput.name = "output";
    for (size_t i = 0; i < outShape.rank(); ++i) {
        qnnOutput.dims.push_back(static_cast<uint32_t>(outShape.dim(i)));
    }
    qnnOutput.dtype = outDtype == tensor::TensorDType::F32
                      ? QnnDataType::Float32 : QnnDataType::Float16;
    qnnOutput.layout = QnnLayout::NCHW;
    qnnOutput.isConst = false;

    // Allocate output buffer
    size_t outSize = QnnTensorAdapter::tensorDataSize(qnnOutput);
    void* outBuf = nullptr;
    if (posix_memalign(&outBuf, 128, outSize) != 0) {
        error = "Failed to allocate NPU output buffer";
        cleanupTempAllocations(tempAllocs);
        return false;
    }
    qnnOutput.dataPtr = outBuf;
    qnnOutput.dataSize = outSize;

#ifdef LOCALIMAGE_QNN
    // Real QNN execution path
    // Build graph
    if (!buildSingleOpGraph(op, attr, qnnInputs, qnnOutput, error)) {
        free(outBuf);
        cleanupTempAllocations(tempAllocs);
        return false;
    }
    // Execute
    if (!executeSingleOpGraph(qnnInputs, qnnOutput, error)) {
        free(outBuf);
        cleanupTempAllocations(tempAllocs);
        return false;
    }
#else
    // Stub path: without QNN SDK, we can't actually execute on NPU
    // Report this clearly so the fallback mechanism works correctly
    (void)outBuf;
    error = "NPU execution unavailable: QNN SDK not compiled in";
    cleanupTempAllocations(tempAllocs);
    return false;
#endif

    // Read output back to LocalImage tensor
    bool readOk = readQnnOutput(qnnOutput, outShape, outDtype, output, error);

    // Cleanup
    free(outBuf);
    cleanupTempAllocations(tempAllocs);

    return readOk;
}

// ============================================================================
// Input preparation
// ============================================================================

bool QnnGraphExecutor::prepareQnnInputs(
    const std::vector<tensor::Tensor>& localInputs,
    std::vector<QnnTensorInfo>& qnnInputs,
    std::vector<void*>& tempAllocs,
    std::string& error) {

    qnnInputs.reserve(localInputs.size());

    for (size_t i = 0; i < localInputs.size(); ++i) {
        const auto& src = localInputs[i];
        QnnTensorInfo info;
        info.name = "input_" + std::to_string(i);

        const bool zeroCopy = QnnTensorAdapter::fromLocalImageTensor(src, info, error);
        if (!src.valid()) return false;

        if (zeroCopy) {
            // Can use data pointer directly
            qnnInputs.push_back(std::move(info));
        } else {
            // Need to repack for NPU layout
            size_t packedSize = 0;
            QnnLayout targetLayout = info.layout;
            void* packed = QnnTensorAdapter::packTensorForNpu(
                src, targetLayout, packedSize, error);
            if (!packed) return false;

            info.dataPtr = packed;
            info.dataSize = packedSize;
            qnnInputs.push_back(std::move(info));
            tempAllocs.push_back(packed);
        }
    }

    return true;
}

bool QnnGraphExecutor::readQnnOutput(const QnnTensorInfo& qnnOutput,
                                      const tensor::TensorShape& expectedShape,
                                      tensor::TensorDType expectedDtype,
                                      tensor::Tensor& output,
                                      std::string& error) {
    tensor::TensorRuntime rt;
    output = rt.createTensor(expectedShape, expectedDtype, error);
    if (!output.valid()) return false;

    if (output.byteSize() != qnnOutput.dataSize) {
        error = "Output size mismatch: expected " + std::to_string(output.byteSize()) +
                " got " + std::to_string(qnnOutput.dataSize);
        return false;
    }

    std::memcpy(output.mutableData(), qnnOutput.dataPtr, qnnOutput.dataSize);
    return true;
}

void QnnGraphExecutor::cleanupTempAllocations(std::vector<void*>& allocs) {
    for (void* p : allocs) {
        if (p) free(p);
    }
    allocs.clear();
}

// ============================================================================
// Output shape inference (mirrors graph_runtime logic)
// ============================================================================

bool QnnGraphExecutor::inferOutputShape(
    ir::Op op,
    const ir::Attributes& attr,
    const std::vector<tensor::Tensor>& inputs,
    tensor::TensorShape& outShape,
    tensor::TensorDType& outDtype,
    std::string& error) {

    if (inputs.empty()) {
        error = "At least one input required for shape inference";
        return false;
    }

    const auto& a = inputs[0];
    outDtype = a.dtype();

    auto broadcast = [&](const tensor::TensorShape& x,
                          const tensor::TensorShape& y,
                          tensor::TensorShape& out) -> bool {
        size_t r = std::max(x.rank(), y.rank());
        std::vector<uint64_t> d(r, 1);
        for (size_t i = 0; i < r; ++i) {
            uint64_t xd = i < x.rank() ? x.dim(x.rank() - 1 - i) : 1;
            uint64_t yd = i < y.rank() ? y.dim(y.rank() - 1 - i) : 1;
            if (xd != yd && xd != 1 && yd != 1) {
                error = "Shape inference broadcast mismatch";
                return false;
            }
            d[r - 1 - i] = std::max(xd, yd);
        }
        out = tensor::TensorShape(std::move(d));
        return out.valid();
    };

    switch(op) {
        case ir::Op::Add:
        case ir::Op::Sub:
        case ir::Op::Mul:
        case ir::Op::Div: {
            if (inputs.size() < 2) { error = "Binary op requires 2 inputs"; return false; }
            return broadcast(a.shape(), inputs[1].shape(), outShape);
        }
        case ir::Op::Exp:
        case ir::Op::Sqrt:
        case ir::Op::Rsqrt:
        case ir::Op::SiLU:
        case ir::Op::GELU:
        case ir::Op::Softmax:
        case ir::Op::LayerNorm:
        case ir::Op::RMSNorm:
        case ir::Op::GroupNorm:
        case ir::Op::Clamp:
            outShape = a.shape();
            return true;
        case ir::Op::Reshape: {
            if (attr.reshape_shape.empty()) {
                error = "Reshape requires reshape_shape";
                return false;
            }
            outShape = tensor::TensorShape(attr.reshape_shape);
            if (!outShape.valid()) {
                error = "Invalid reshape shape";
                return false;
            }
            return true;
        }
        case ir::Op::Broadcast: {
            if (attr.broadcast_shape.empty()) {
                error = "Broadcast requires broadcast_shape";
                return false;
            }
            outShape = tensor::TensorShape(attr.broadcast_shape);
            return outShape.valid();
        }
        case ir::Op::Slice: {
            if (attr.slice_starts.size() != a.shape().rank() ||
                attr.slice_lengths.size() != a.shape().rank()) {
                error = "Slice requires per-dimension starts and lengths";
                return false;
            }
            auto d = a.shape().dims();
            for (size_t i = 0; i < d.size(); ++i) {
                if (attr.slice_starts[i] > d[i] ||
                    attr.slice_lengths[i] > d[i] - attr.slice_starts[i]) {
                    error = "Slice range out of bounds";
                    return false;
                }
                d[i] = attr.slice_lengths[i];
            }
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Transpose: {
            if (attr.permutation.size() != a.shape().rank()) {
                error = "Transpose permutation rank mismatch";
                return false;
            }
            std::vector<uint64_t> d(a.shape().rank());
            for (size_t i = 0; i < attr.permutation.size(); ++i) {
                d[i] = a.shape().dim(attr.permutation[i]);
            }
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Concat: {
            if (inputs.size() < 2 || attr.axes.empty()) {
                error = "Concat requires at least 2 inputs and axis";
                return false;
            }
            size_t ax = attr.axes[0];
            auto d = a.shape().dims();
            uint64_t sum = 0;
            for (const auto& in : inputs) {
                if (in.shape().rank() != a.shape().rank()) {
                    error = "Concat rank mismatch";
                    return false;
                }
                sum += in.shape().dim(ax);
            }
            d[ax] = sum;
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::MatMul:
        case ir::Op::BatchedMatMul: {
            if (inputs.size() < 2) { error = "MatMul requires 2 inputs"; return false; }
            const auto& b = inputs[1];
            if (a.shape().rank() != b.shape().rank() || a.shape().rank() < 2) {
                error = "MatMul shape rank mismatch";
                return false;
            }
            if (a.shape().dim(a.shape().rank() - 1) != b.shape().dim(b.shape().rank() - 2)) {
                error = "MatMul inner dimension mismatch";
                return false;
            }
            auto d = a.shape().dims();
            d.back() = b.shape().dim(b.shape().rank() - 1);
            for (size_t i = 0; i + 2 < d.size(); ++i) {
                auto x = a.shape().dim(i);
                auto y = b.shape().dim(i);
                d[i] = std::max(x, y);
            }
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Linear: {
            if (inputs.size() < 2) { error = "Linear requires input and weight"; return false; }
            const auto& w = inputs[1];
            if (w.shape().rank() != 2 || a.shape().rank() < 2) {
                error = "Linear weight must be rank-2";
                return false;
            }
            if (a.shape().dim(a.shape().rank() - 1) != w.shape().dim(0)) {
                error = "Linear input/weight dimension mismatch";
                return false;
            }
            auto d = a.shape().dims();
            d.back() = w.shape().dim(1);
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Conv2D: {
            if (inputs.size() < 2) { error = "Conv2D requires input and weight"; return false; }
            const auto& w = inputs[1];
            if (a.shape().rank() != 4 || w.shape().rank() != 4) {
                error = "Conv2D requires rank-4 input and weight";
                return false;
            }
            uint64_t kh = w.shape().dim(2);
            uint64_t kw = w.shape().dim(3);
            uint64_t effh = 1 + static_cast<uint64_t>(attr.dilation) * (kh - 1);
            uint64_t effw = 1 + static_cast<uint64_t>(attr.dilation) * (kw - 1);
            uint64_t nh = a.shape().dim(2) + 2 * static_cast<uint64_t>(attr.padding);
            uint64_t nw = a.shape().dim(3) + 2 * static_cast<uint64_t>(attr.padding);
            if (nh < effh || nw < effw) {
                error = "Conv2D kernel exceeds padded input";
                return false;
            }
            auto d = a.shape().dims();
            d[1] = w.shape().dim(0);
            d[2] = (nh - effh) / static_cast<uint64_t>(attr.stride) + 1;
            d[3] = (nw - effw) / static_cast<uint64_t>(attr.stride) + 1;
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Upsample: {
            if (a.shape().rank() != 4 || attr.scale_factor == 0) {
                error = "Upsample requires rank-4 and positive scale";
                return false;
            }
            auto d = a.shape().dims();
            d[2] *= attr.scale_factor;
            d[3] *= attr.scale_factor;
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::Attention: {
            if (inputs.size() < 3) { error = "Attention requires Q, K, V"; return false; }
            const auto& k = inputs[1];
            const auto& v = inputs[2];
            if (a.shape().rank() != k.shape().rank() ||
                a.shape().rank() != v.shape().rank()) {
                error = "Attention rank mismatch";
                return false;
            }
            if (a.shape().dim(a.shape().rank() - 1) != k.shape().dim(k.shape().rank() - 1)) {
                error = "Attention Q/K head_dim mismatch";
                return false;
            }
            if (k.shape().dim(k.shape().rank() - 2) != v.shape().dim(v.shape().rank() - 2)) {
                error = "Attention K/V seq_len mismatch";
                return false;
            }
            auto d = a.shape().dims();
            d.back() = v.shape().dim(v.shape().rank() - 1);
            outShape = tensor::TensorShape(std::move(d));
            return true;
        }
        case ir::Op::RoPE: {
            if (inputs.size() < 2) { error = "RoPE requires input and freq"; return false; }
            outShape = a.shape();
            return true;
        }
        default:
            error = "Shape inference not implemented for op: " +
                    std::string(ir::opName(op));
            return false;
    }
}

// ============================================================================
// QNN graph building (stub when SDK not available)
// ============================================================================

#ifdef LOCALIMAGE_QNN

bool QnnGraphExecutor::buildSingleOpGraph(ir::Op op,
                                           const ir::Attributes& attr,
                                           const std::vector<QnnTensorInfo>& inputInfos,
                                           const QnnTensorInfo& outputInfo,
                                           std::string& error) {
    // Get QNN interface from the loaded backend library
    QnnInterface_t* qnnIface = getQnnInterface(error);
    if (!qnnIface || !qnnIface->graph) {
        if (error.empty()) error = "QNN graph interface not available";
        return false;
    }

    // Map LocalImage IR op to QNN op type
    OpMappingResult mapping = opMapper_.mapOp(op, attr);
    if (!mapping.supported) {
        error = "Unsupported op for QNN graph build: " + std::string(ir::opName(op));
        return false;
    }

    const char* qnnOpType = toQnnOpTypeName(mapping.targetOp);
    if (!qnnOpType) {
        error = "No QNN op type name mapping for op: " +
                std::string(ir::opName(op));
        return false;
    }

    // Compute cache key from op + input/output tensor signatures
    const std::string key = graphCacheKey(
        OpMapper::qnnOpName(mapping.targetOp), inputInfos, outputInfo);

    // Check cache: if graph already built for this config, reuse it
    auto cacheIt = opCache_.find(key);
    if (cacheIt != opCache_.end() && cacheIt->second.graphHandle) {
        return true; // graph already built and cached
    }

    // ========================================================================
    // Step 1: Create QNN graph
    // ========================================================================
    QnnGraph_Handle_t graphHandle = nullptr;
    Qnn_ErrorHandle_t status;

    // Graph configs: standard QNN tagged-array pattern
    std::vector<QnnGraph_Config_t> graphConfigs;
    {
        QnnGraph_Config_t cfg;
        cfg.option = QNN_GRAPH_CONFIG_OPTION_END;
        graphConfigs.push_back(cfg);
    }

    status = qnnIface->graph->create(
        context_ ? context_->contextHandle() : nullptr,
        graphConfigs.data(),
        static_cast<uint32_t>(graphConfigs.size()),
        &graphHandle);

    if (status != QNN_SUCCESS || !graphHandle) {
        error = "Failed to create QNN graph. Error code: " +
                std::to_string(status);
        return false;
    }

    // ========================================================================
    // Step 2: Build tensor descriptors for inputs and output
    // ========================================================================
    // We need Qnn_Tensor_t objects that remain valid for the duration of
    // the addNode call. We use stack-allocated vectors of Qnn_Tensor_t.
    std::vector<Qnn_Tensor_t> inputTensors;
    inputTensors.reserve(inputInfos.size());
    for (const auto& inInfo : inputInfos) {
        Qnn_Tensor_t t{};
        fillQnnTensorDesc(t, inInfo, QNN_TENSOR_TYPE_INPUT);
        inputTensors.push_back(t);
    }

    Qnn_Tensor_t outputTensor{};
    fillQnnTensorDesc(outputTensor, outputInfo, QNN_TENSOR_TYPE_OUTPUT);

    // ========================================================================
    // Step 3: Build node config and add node to graph
    // ========================================================================
    QnnGraph_NodeConfig_t nodeConfig{};
    nodeConfig.name = "single_op_node";
    nodeConfig.type = qnnOpType;
    nodeConfig.inputTensors = inputTensors.data();
    nodeConfig.numInputTensors = static_cast<uint32_t>(inputTensors.size());
    nodeConfig.outputTensors = &outputTensor;
    nodeConfig.numOutputTensors = 1;

    // Op-specific configuration
    // Different op families require different config structs.
    // We allocate op-specific configs on the stack and point nodeConfig to them.
    switch(mapping.targetOp) {
        case QnnOpType::MatMul:
        case QnnOpType::BatchMatMul: {
            auto mmCfg = OpConfigBuilder::buildMatMul(attr);
            QnnGraph_MatMulNodeConfig_t matMulConfig{};
            matMulConfig.transposeA = mmCfg.transposeA;
            matMulConfig.transposeB = mmCfg.transposeB;
            nodeConfig.nodeConfig = &matMulConfig;
            break;
        }
        case QnnOpType::Conv2D:
        case QnnOpType::DepthwiseConv2D: {
            auto convCfg = OpConfigBuilder::buildConv2D(attr);
            QnnGraph_Conv2dNodeConfig_t convConfig{};
            convConfig.strideHeight = convCfg.strideH;
            convConfig.strideWidth = convCfg.strideW;
            convConfig.padHeight = convCfg.padH;
            convConfig.padWidth = convCfg.padW;
            convConfig.dilationHeight = convCfg.dilationH;
            convConfig.dilationWidth = convCfg.dilationW;
            convConfig.groups = convCfg.groups;
            nodeConfig.nodeConfig = &convConfig;
            break;
        }
        case QnnOpType::Softmax: {
            auto smCfg = OpConfigBuilder::buildSoftmax(attr);
            QnnGraph_SoftmaxNodeConfig_t softmaxConfig{};
            softmaxConfig.axis = smCfg.axis;
            nodeConfig.nodeConfig = &softmaxConfig;
            break;
        }
        case QnnOpType::LayerNorm: {
            auto lnCfg = OpConfigBuilder::buildLayerNorm(attr);
            QnnGraph_LayerNormNodeConfig_t lnConfig{};
            lnConfig.epsilon = lnCfg.epsilon;
            lnConfig.axis = lnCfg.axis;
            nodeConfig.nodeConfig = &lnConfig;
            break;
        }
        case QnnOpType::GroupNorm: {
            auto gnCfg = OpConfigBuilder::buildGroupNorm(attr);
            QnnGraph_GroupNormNodeConfig_t gnConfig{};
            gnConfig.epsilon = gnCfg.epsilon;
            gnConfig.groups = gnCfg.groups;
            nodeConfig.nodeConfig = &gnConfig;
            break;
        }
        case QnnOpType::Clamp: {
            QnnGraph_ClampNodeConfig_t clampConfig{};
            clampConfig.minValue = attr.clamp_min;
            clampConfig.maxValue = attr.clamp_max;
            nodeConfig.nodeConfig = &clampConfig;
            break;
        }
        case QnnOpType::Transpose: {
            auto tpCfg = OpConfigBuilder::buildTranspose(attr);
            QnnGraph_TransposeNodeConfig_t tpConfig{};
            tpConfig.perm = tpCfg.perm.data();
            tpConfig.numPerms = static_cast<uint32_t>(tpCfg.perm.size());
            nodeConfig.nodeConfig = &tpConfig;
            break;
        }
        case QnnOpType::Slice: {
            uint32_t rank = static_cast<uint32_t>(
                !inputInfos.empty() ? inputInfos[0].dims.size() : 4);
            auto slCfg = OpConfigBuilder::buildSlice(attr, rank);
            QnnGraph_SliceNodeConfig_t sliceConfig{};
            sliceConfig.starts = slCfg.starts.data();
            sliceConfig.sizes = slCfg.lengths.data();
            sliceConfig.rank = rank;
            nodeConfig.nodeConfig = &sliceConfig;
            break;
        }
        case QnnOpType::Concat: {
            auto ccCfg = OpConfigBuilder::buildConcat(attr);
            QnnGraph_ConcatNodeConfig_t concatConfig{};
            concatConfig.axis = ccCfg.axis;
            nodeConfig.nodeConfig = &concatConfig;
            break;
        }
        case QnnOpType::UpsampleNearest:
        case QnnOpType::UpsampleBilinear: {
            auto rsCfg = OpConfigBuilder::buildUpsample(attr);
            QnnGraph_ResizeNodeConfig_t resizeConfig{};
            resizeConfig.scaleFactor = rsCfg.scaleFactor;
            nodeConfig.nodeConfig = &resizeConfig;
            break;
        }
        // Elementwise ops (Add, Sub, Mul, Div), activation ops (ReLU, SiLU,
        // GELU, Sigmoid, Tanh, Exp, Sqrt, Rsqrt), and reshape/squeeze/unsqueeze
        // do not require additional config parameters beyond input/output tensors.
        default:
            nodeConfig.nodeConfig = nullptr;
            break;
    }

    status = qnnIface->graph->addNode(graphHandle, &nodeConfig);
    if (status != QNN_SUCCESS) {
        error = "Failed to add node '" + std::string(qnnOpType) +
                "' to QNN graph. Error code: " + std::to_string(status);
        // Clean up the partially created graph
        qnnIface->graph->free(graphHandle);
        return false;
    }

    // ========================================================================
    // Step 4: Finalize the graph
    // ========================================================================
    // After finalize, the graph is ready for execution. No more nodes
    // or tensors can be added.
    status = qnnIface->graph->finalize(graphHandle);
    if (status != QNN_SUCCESS) {
        error = "Failed to finalize QNN graph. Error code: " +
                std::to_string(status);
        qnnIface->graph->free(graphHandle);
        return false;
    }

    // ========================================================================
    // Step 5: Cache the graph handle
    // ========================================================================
    SingleOpCache cacheEntry;
    cacheEntry.opName = OpMapper::qnnOpName(mapping.targetOp);
    cacheEntry.dtype = outputInfo.dtype;
    cacheEntry.inputCount = static_cast<uint32_t>(inputInfos.size());
    cacheEntry.graphHandle = graphHandle;

    opCache_[key] = std::move(cacheEntry);

    return true;
}

bool QnnGraphExecutor::executeSingleOpGraph(const std::vector<QnnTensorInfo>& inputs,
                                             QnnTensorInfo& output,
                                             std::string& error) {
    // Get QNN interface
    QnnInterface_t* qnnIface = getQnnInterface(error);
    if (!qnnIface || !qnnIface->graph) {
        if (error.empty()) error = "QNN graph interface not available for execution";
        return false;
    }

    // ========================================================================
    // Find the cached graph handle
    // ========================================================================
    // Since executeSingleOpGraph does not receive the op type directly,
    // we search the cache for a graph matching the input/output tensor
    // signature. This works because buildSingleOpGraph is always called
    // immediately before executeSingleOpGraph with the same tensor shapes.
    QnnGraph_Handle_t graphHandle = nullptr;
    std::string foundKey;

    for (const auto& [key, cache] : opCache_) {
        if (cache.graphHandle && cache.inputCount == inputs.size()) {
            // Additional check: verify output dtype matches
            if (cache.dtype == output.dtype) {
                graphHandle = cache.graphHandle;
                foundKey = key;
                break; // take first matching graph (most recently built)
            }
        }
    }

    if (!graphHandle) {
        error = "No cached QNN graph found for execution. "
                "Graph must be built before execute.";
        return false;
    }

    // ========================================================================
    // Step 1: Prepare input tensors with data pointers
    // ========================================================================
    std::vector<Qnn_Tensor_t> execInputs;
    execInputs.reserve(inputs.size());
    for (const auto& inInfo : inputs) {
        Qnn_Tensor_t t{};
        fillQnnTensorExec(t, inInfo, QNN_TENSOR_TYPE_INPUT);
        execInputs.push_back(t);
    }

    // Step 2: Prepare output tensor with data buffer
    Qnn_Tensor_t execOutput{};
    fillQnnTensorExec(execOutput, output, QNN_TENSOR_TYPE_OUTPUT);

    // ========================================================================
    // Step 3: Execute the graph
    // ========================================================================
    // graph->execute() signature:
    //   Qnn_ErrorHandle_t (*execute)(
    //       QnnGraph_Handle_t graph,
    //       const Qnn_Tensor_t* inputTensors,
    //       uint32_t numInputTensors,
    //       Qnn_Tensor_t* outputTensors,
    //       uint32_t numOutputTensors,
    //       Qnn_SignalHandle_t signal);
    //
    // signal is nullptr for synchronous blocking execution.

    Qnn_ErrorHandle_t status = qnnIface->graph->execute(
        graphHandle,
        execInputs.data(),
        static_cast<uint32_t>(execInputs.size()),
        &execOutput,
        1,  // numOutputTensors
        nullptr); // signal = nullptr -> synchronous execution

    if (status != QNN_SUCCESS) {
        error = "QNN graph execution failed. Error code: " +
                std::to_string(status);
        return false;
    }

    // ========================================================================
    // Step 4: Output data is already in the buffer we provided
    // ========================================================================
    // The output.dataPtr was set by the caller before execute.
    // QNN writes output directly into our buffer.
    // output.dataSize is already correct from the caller's allocation.

    return true;
}

#else

bool QnnGraphExecutor::buildSingleOpGraph(ir::Op /*op*/,
                                           const ir::Attributes& /*attr*/,
                                           const std::vector<QnnTensorInfo>& /*inputInfos*/,
                                           const QnnTensorInfo& /*outputInfo*/,
                                           std::string& error) {
    error = "QNN SDK not available for graph construction";
    return false;
}

bool QnnGraphExecutor::executeSingleOpGraph(const std::vector<QnnTensorInfo>& /*inputs*/,
                                             QnnTensorInfo& /*output*/,
                                             std::string& error) {
    error = "QNN SDK not available for graph execution";
    return false;
}

#endif

// ============================================================================
// Subgraph execution (for fused op optimization)
// ============================================================================

bool QnnGraphExecutor::executeSubgraph(
    const std::vector<ir::Node>& /*nodes*/,
    const std::unordered_map<std::string, tensor::Tensor>& /*inputs*/,
    std::unordered_map<std::string, tensor::Tensor>& /*outputs*/,
    std::string& error) {
    // Subgraph execution fuses multiple ops into a single QNN graph
    // for better performance (less dispatch overhead, more fusion opportunities)
    //
    // Implementation plan:
    // 1. Build QNN graph with all nodes in the subgraph
    // 2. Register all inputs and outputs
    // 3. Execute once
    // 4. Read all outputs
    //
    // For now, this is a placeholder - per-op execution is the primary path
    error = "Subgraph execution not yet implemented (use per-op mode)";
    return false;
}

} // namespace qnn
} // namespace npu
} // namespace localimage
