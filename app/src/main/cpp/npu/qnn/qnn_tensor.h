#pragma once

#include "qnn_types.h"
#include "../../tensor/tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace localimage {
namespace npu {
namespace qnn {

// Tensor adapter: converts between LocalImage Tensor and QNN Tensor format
// Key design principle:
// - Zero-copy when layouts match (NCHW F16 -> NCHW F16)
// - On-the-fly conversion when layouts differ (cost is per-op, not per-model)
// - Weight tensors can be pre-packaged and cached for repeated use
class QnnTensorAdapter {
public:
    QnnTensorAdapter();
    ~QnnTensorAdapter();

    // Convert a LocalImage tensor to QNN tensor info
    // Returns true if zero-copy is possible (no layout/dtype conversion needed)
    static bool fromLocalImageTensor(const tensor::Tensor& src,
                                      QnnTensorInfo& dst,
                                      std::string& error);

    // Convert QNN tensor output back to LocalImage tensor
    static bool toLocalImageTensor(const QnnTensorInfo& src,
                                    tensor::Tensor& dst,
                                    std::string& error);

    // Check if a tensor can be used directly with NPU (zero-copy)
    static bool isZeroCopyCompatible(const tensor::Tensor& t);

    // Get preferred NPU layout for a given op and input rank
    static QnnLayout preferredLayout(QnnOpType op, uint32_t rank);

    // Calculate QNN tensor data size in bytes
    static size_t tensorDataSize(const QnnTensorInfo& info);

    // Convert tensor layout (repack data)
    // Caller owns the returned buffer and must free it with freePackedData()
    static void* packTensorForNpu(const tensor::Tensor& src,
                                  QnnLayout targetLayout,
                                  size_t& outSize,
                                  std::string& error);

    static void freePackedData(void* data);

    // Get size of QNN data type
    static size_t dtypeSize(QnnDataType dtype);
};

// Weight cache entry: pre-packed weight tensor for NPU
// Stored on disk with model_hash + dsp_version as part of cache key
struct WeightCacheEntry {
    std::string tensorName;
    QnnDataType dtype;
    QnnLayout layout;
    std::vector<uint32_t> dims;
    std::vector<uint8_t> packedData;
    uint64_t crc64 = 0;
};

// Weight pre-packer: converts model weights to NPU-optimized layout
// Results are cached to disk for fast reload
class WeightPrepacker {
public:
    WeightPrepacker();
    ~WeightPrepacker();

    // Set cache directory
    void setCacheDir(const std::string& dir) { cacheDir_ = dir; }

    // Pre-pack a single weight tensor
    // Returns cached (pre-packed) data pointer and size
    // If cache miss: packs now, writes to cache, returns result
    // If cache hit: mmap's cached file, returns pointer
    bool prepackWeight(const std::string& tensorName,
                       const tensor::Tensor& weight,
                       const std::string& modelHash,
                       DspVersion dspVersion,
                       WeightCacheEntry& outEntry,
                       std::string& error);

    // Check if a weight is already cached
    bool isCached(const std::string& tensorName,
                  const std::string& modelHash,
                  DspVersion dspVersion) const;

    // Clear all cached weights for a model
    bool clearModelCache(const std::string& modelHash, std::string& error);

private:
    std::string cacheDir_;
    std::string cacheKey(const std::string& tensorName,
                         const std::string& modelHash,
                         DspVersion dspVersion) const;
};

} // namespace qnn
} // namespace npu
} // namespace localimage
