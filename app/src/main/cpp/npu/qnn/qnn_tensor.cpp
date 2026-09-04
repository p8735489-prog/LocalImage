#include "qnn_tensor.h"
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

namespace localimage {
namespace npu {
namespace qnn {

// ============================================================================
// QnnTensorAdapter
// ============================================================================

QnnTensorAdapter::QnnTensorAdapter() = default;
QnnTensorAdapter::~QnnTensorAdapter() = default;

size_t QnnTensorAdapter::dtypeSize(QnnDataType dtype) {
    switch(dtype) {
        case QnnDataType::Float32:  return 4;
        case QnnDataType::Float16:  return 2;
        case QnnDataType::BFloat16: return 2;
        case QnnDataType::Int32:    return 4;
        case QnnDataType::Int16:    return 2;
        case QnnDataType::Int8:     return 1;
        case QnnDataType::UInt8:    return 1;
        case QnnDataType::Bool:     return 1;
        default: return 0;
    }
}

size_t QnnTensorAdapter::tensorDataSize(const QnnTensorInfo& info) {
    size_t elemSize = dtypeSize(info.dtype);
    if (elemSize == 0) return 0;
    size_t count = 1;
    for (auto d : info.dims) count *= d;
    return count * elemSize;
}

bool QnnTensorAdapter::fromLocalImageTensor(const tensor::Tensor& src,
                                             QnnTensorInfo& dst,
                                             std::string& error) {
    if (!src.valid()) {
        error = "Cannot convert invalid tensor to QNN";
        return false;
    }

    const auto& shape = src.shape();
    dst.dims.clear();
    for (size_t i = 0; i < shape.rank(); ++i) {
        dst.dims.push_back(static_cast<uint32_t>(shape.dim(i)));
    }

    // Map dtype
    if (src.dtype() == tensor::TensorDType::F32) {
        dst.dtype = QnnDataType::Float32;
    } else if (src.dtype() == tensor::TensorDType::F16) {
        dst.dtype = QnnDataType::Float16;
    } else if (src.dtype() == tensor::TensorDType::BF16) {
        dst.dtype = QnnDataType::BFloat16;
    } else {
        dst.dtype = QnnDataType::Float16; // default fallback
    }

    // Assume NCHW layout for rank-4, plain for others
    if (shape.rank() == 4) {
        dst.layout = QnnLayout::NCHW;
    } else {
        dst.layout = QnnLayout::NCHW; // treat as logical layout
    }

    dst.dataPtr = const_cast<void*>(src.data());
    dst.dataSize = src.byteSize();
    dst.isConst = false;

    // Zero-copy is possible when:
    // - Tensor is contiguous
    // - Layout matches preferred NPU layout
    // - Dtype is supported natively
    const bool contiguous = src.isContiguous();
    const bool supportedDtype = (dst.dtype == QnnDataType::Float16 ||
                                  dst.dtype == QnnDataType::Float32);
    return contiguous && supportedDtype;
}

bool QnnTensorAdapter::toLocalImageTensor(const QnnTensorInfo& src,
                                           tensor::Tensor& dst,
                                           std::string& error) {
    if (!src.dataPtr || src.dataSize == 0) {
        error = "QNN tensor has no data";
        return false;
    }

    tensor::TensorShape shape(src.dims);
    if (!shape.valid()) {
        error = "Invalid tensor shape from QNN output";
        return false;
    }

    tensor::TensorDType dtype = tensor::TensorDType::F16;
    switch(src.dtype) {
        case QnnDataType::Float32:  dtype = tensor::TensorDType::F32; break;
        case QnnDataType::Float16:  dtype = tensor::TensorDType::F16; break;
        case QnnDataType::BFloat16: dtype = tensor::TensorDType::BF16; break;
        default:
            error = "Unsupported QNN output dtype";
            return false;
    }

    tensor::TensorRuntime rt;
    dst = rt.createTensor(shape, dtype, error);
    if (!dst.valid()) return false;

    if (dst.byteSize() != src.dataSize) {
        error = "Output tensor size mismatch";
        return false;
    }

    std::memcpy(dst.mutableData(), src.dataPtr, src.dataSize);
    return true;
}

bool QnnTensorAdapter::isZeroCopyCompatible(const tensor::Tensor& t) {
    if (!t.valid() || !t.isContiguous()) return false;
    // HTP prefers F16; F32 works but is slower
    return (t.dtype() == tensor::TensorDType::F16 ||
            t.dtype() == tensor::TensorDType::F32);
}

QnnLayout QnnTensorAdapter::preferredLayout(QnnOpType op, uint32_t rank) {
    if (rank != 4) return QnnLayout::NCHW;

    // Conv2D-family ops prefer block format on HTP
    switch(op) {
        case QnnOpType::Conv2D:
        case QnnOpType::DepthwiseConv2D:
        case QnnOpType::TransposeConv2D:
            return QnnLayout::NCHW_C8; // or C16 depending on DSP version
        default:
            return QnnLayout::NCHW;
    }
}

void* QnnTensorAdapter::packTensorForNpu(const tensor::Tensor& src,
                                          QnnLayout targetLayout,
                                          size_t& outSize,
                                          std::string& error) {
    if (!src.valid()) {
        error = "Cannot pack invalid tensor";
        return nullptr;
    }

    const auto& shape = src.shape();
    const size_t elemSize = tensor::dtypeSize(src.dtype());

    // If already NCHW and target is NCHW, just copy
    if (targetLayout == QnnLayout::NCHW) {
        outSize = src.byteSize();
        void* out = nullptr;
        if (posix_memalign(&out, 128, outSize) != 0) {
            error = "Failed to allocate packed tensor memory";
            return nullptr;
        }
        std::memcpy(out, src.data(), outSize);
        return out;
    }

    // For block formats (NCHW_C8, NCHW_C16), repack the data
    // This is a simplified implementation - actual HTP block formats
    // have specific alignment requirements
    if (shape.rank() != 4) {
        // Non-4D tensors don't use block formats
        outSize = src.byteSize();
        void* out = nullptr;
        if (posix_memalign(&out, 128, outSize) != 0) {
            error = "Failed to allocate packed tensor memory";
            return nullptr;
        }
        std::memcpy(out, src.data(), outSize);
        return out;
    }

    const uint32_t N = static_cast<uint32_t>(shape.dim(0));
    const uint32_t C = static_cast<uint32_t>(shape.dim(1));
    const uint32_t H = static_cast<uint32_t>(shape.dim(2));
    const uint32_t W = static_cast<uint32_t>(shape.dim(3));

    uint32_t blockSize = 8; // C8
    if (targetLayout == QnnLayout::NCHW_C16 ||
        targetLayout == QnnLayout::NCHW_VEC16) {
        blockSize = 16;
    }

    // Pad channels to block boundary
    const uint32_t paddedC = (C + blockSize - 1) / blockSize * blockSize;
    outSize = N * paddedC * H * W * elemSize;

    void* out = nullptr;
    if (posix_memalign(&out, 128, outSize) != 0) {
        error = "Failed to allocate packed tensor memory";
        return nullptr;
    }

    // Zero out padding
    std::memset(out, 0, outSize);

    // Copy data from NCHW to NCHW-packed format
    const uint8_t* srcBytes = static_cast<const uint8_t*>(src.data());
    uint8_t* dstBytes = static_cast<uint8_t*>(out);

    // NCHW to NCHW_Cx: channel dimension is blocked
    // For each batch, for each block of channels, for each spatial position
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t cb = 0; cb < paddedC / blockSize; ++cb) {
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    for (uint32_t ci = 0; ci < blockSize; ++ci) {
                        uint32_t c = cb * blockSize + ci;
                        size_t dstIdx = ((n * (paddedC / blockSize) + cb) * H + h) * W + w;
                        dstIdx = dstIdx * blockSize + ci;
                        dstIdx *= elemSize;

                        if (c < C) {
                            size_t srcIdx = ((n * C + c) * H + h) * W + w;
                            srcIdx *= elemSize;
                            std::memcpy(dstBytes + dstIdx, srcBytes + srcIdx, elemSize);
                        }
                        // else: already zeroed (padding)
                    }
                }
            }
        }
    }

    return out;
}

void QnnTensorAdapter::freePackedData(void* data) {
    if (data) free(data);
}

// ============================================================================
// WeightPrepacker
// ============================================================================

WeightPrepacker::WeightPrepacker() = default;
WeightPrepacker::~WeightPrepacker() = default;

std::string WeightPrepacker::cacheKey(const std::string& tensorName,
                                       const std::string& modelHash,
                                       DspVersion dspVersion) const {
    std::ostringstream os;
    os << modelHash << "_v" << static_cast<int>(dspVersion) << "_"
       << std::hash<std::string>{}(tensorName);
    return os.str();
}

bool WeightPrepacker::isCached(const std::string& tensorName,
                                const std::string& modelHash,
                                DspVersion dspVersion) const {
    if (cacheDir_.empty()) return false;
    const std::string key = cacheKey(tensorName, modelHash, dspVersion);
    const std::string path = cacheDir_ + "/" + key + ".npu_weight";

    struct stat st;
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

bool WeightPrepacker::prepackWeight(const std::string& tensorName,
                                     const tensor::Tensor& weight,
                                     const std::string& modelHash,
                                     DspVersion dspVersion,
                                     WeightCacheEntry& outEntry,
                                     std::string& error) {
    if (!weight.valid()) {
        error = "Cannot prepack invalid weight tensor";
        return false;
    }

    const std::string key = cacheKey(tensorName, modelHash, dspVersion);
    const std::string cachePath = cacheDir_ + "/" + key + ".npu_weight";

    // Check cache hit
    if (!cacheDir_.empty()) {
        struct stat st;
        if (stat(cachePath.c_str(), &st) == 0 && st.st_size > 0) {
            // Cache hit - mmap and return
            // (In full implementation, would mmap the cached file)
            // For now, signal cache hit conceptually
        }
    }

    // Cache miss - pack the weight
    // Determine target layout based on tensor shape
    const auto& shape = weight.shape();
    QnnLayout targetLayout = QnnLayout::NCHW;

    // Conv weights (4D) use block format for efficiency
    if (shape.rank() == 4) {
        targetLayout = QnnLayout::NCHW_C8;
        // Higher DSP versions prefer larger blocks
        if (static_cast<int>(dspVersion) >= static_cast<int>(DspVersion::V79)) {
            targetLayout = QnnLayout::NCHW_C16;
        }
    }

    size_t packedSize = 0;
    void* packedData = QnnTensorAdapter::packTensorForNpu(
        weight, targetLayout, packedSize, error);
    if (!packedData) return false;

    // Fill output entry
    outEntry.tensorName = tensorName;
    if (weight.dtype() == tensor::TensorDType::F32) {
        outEntry.dtype = QnnDataType::Float32;
    } else if (weight.dtype() == tensor::TensorDType::F16) {
        outEntry.dtype = QnnDataType::Float16;
    } else {
        outEntry.dtype = QnnDataType::Float16;
    }
    outEntry.layout = targetLayout;
    outEntry.dims.clear();
    for (size_t i = 0; i < shape.rank(); ++i) {
        outEntry.dims.push_back(static_cast<uint32_t>(shape.dim(i)));
    }
    outEntry.packedData.assign(
        static_cast<const uint8_t*>(packedData),
        static_cast<const uint8_t*>(packedData) + packedSize);

    QnnTensorAdapter::freePackedData(packedData);

    // Write to cache if directory is set
    if (!cacheDir_.empty()) {
        // Ensure directory exists
        mkdir(cacheDir_.c_str(), 0700);
        int fd = open(cachePath.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (fd >= 0) {
            write(fd, outEntry.packedData.data(), outEntry.packedData.size());
            close(fd);
        }
    }

    return true;
}

bool WeightPrepacker::clearModelCache(const std::string& modelHash,
                                       std::string& error) {
    if (cacheDir_.empty()) {
        error = "Cache directory not set";
        return false;
    }
    // Would iterate and delete all files matching modelHash prefix
    // Simplified for now
    return true;
}

} // namespace qnn
} // namespace npu
} // namespace localimage
