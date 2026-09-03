#pragma once

#include "../safetensors/safe_tensor_file.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace localimage::tensor {

enum class TensorDType { F32, F16, BF16, I8, U8, Unknown };
enum class TensorDevice { CPU, MAPPED, GPU, NPU };

const char* dtypeName(TensorDType dtype);
size_t dtypeSize(TensorDType dtype);
TensorDType fromSafeTensorDType(safetensors::DType dtype);

class TensorShape {
public:
    TensorShape() = default;
    explicit TensorShape(std::vector<uint64_t> dims);
    const std::vector<uint64_t>& dims() const { return dims_; }
    size_t rank() const { return dims_.size(); }
    uint64_t dim(size_t index) const;
    uint64_t elementCount() const { return element_count_; }
    bool valid() const { return valid_; }
    const std::string& error() const { return error_; }
private:
    std::vector<uint64_t> dims_;
    uint64_t element_count_ = 1;
    bool valid_ = true;
    std::string error_;
};

class TensorStride {
public:
    TensorStride() = default;
    explicit TensorStride(std::vector<uint64_t> values) : values_(std::move(values)) {}
    static TensorStride contiguous(const TensorShape& shape);
    const std::vector<uint64_t>& values() const { return values_; }
    uint64_t stride(size_t dimension) const;
    bool isContiguous(const TensorShape& shape) const;
private:
    std::vector<uint64_t> values_;
};

class TensorStorage {
public:
    ~TensorStorage();
    static std::shared_ptr<TensorStorage> allocate(size_t bytes, size_t alignment, std::string& error);
    static std::shared_ptr<TensorStorage> mapped(std::shared_ptr<const safetensors::MappedFile> owner);
    const void* data() const { return data_; }
    void* mutableData() { return owned_ ? data_ : nullptr; }
    size_t size() const { return size_; }
    bool owned() const { return owned_; }
    std::shared_ptr<const safetensors::MappedFile> mappedOwner() const { return mapped_owner_; }
private:
    TensorStorage() = default;
    void* data_ = nullptr;
    size_t size_ = 0;
    bool owned_ = false;
    std::shared_ptr<const safetensors::MappedFile> mapped_owner_;
};

class Tensor {
public:
    Tensor() = default;
    static bool create(const TensorShape& shape, TensorDType dtype, TensorDevice device, Tensor& out, std::string& error);
    static Tensor fromView(const safetensors::TensorView& view, std::string& error);

    const TensorShape& shape() const { return shape_; }
    const TensorStride& stride() const { return stride_; }
    TensorDType dtype() const { return dtype_; }
    TensorDevice device() const { return device_; }
    size_t byteSize() const { return byte_size_; }
    bool isContiguous() const { return stride_.isContiguous(shape_); }
    const void* data() const;
    void* mutableData();
    float readFloat32(uint64_t index, std::string& error) const;
    bool writeFloat32(uint64_t index, float value, std::string& error);
    bool valid() const { return storage_ != nullptr && (data_ != nullptr || byte_size_ == 0); }
    std::string reshape(const TensorShape& shape, Tensor& out) const;
    std::string slice(size_t dimension, uint64_t start, uint64_t length, Tensor& out) const;
private:
    std::shared_ptr<TensorStorage> storage_;
    const uint8_t* data_ = nullptr;
    TensorShape shape_;
    TensorStride stride_;
    TensorDType dtype_ = TensorDType::Unknown;
    TensorDevice device_ = TensorDevice::CPU;
    size_t byte_size_ = 0;
};

class TensorAllocator {
public:
    Tensor allocate(const TensorShape& shape, TensorDType dtype, std::string& error) const;
    void release(Tensor& tensor) const { tensor = Tensor{}; }
};

class TensorRuntime {
public:
    Tensor createTensor(const TensorShape& shape, TensorDType dtype, std::string& error) const;
    Tensor createView(const safetensors::TensorView& view, std::string& error) const;
    bool reshape(const Tensor& input, const TensorShape& shape, Tensor& out, std::string& error) const;
    bool slice(const Tensor& input, size_t dimension, uint64_t start, uint64_t length, Tensor& out, std::string& error) const;
    bool transpose(const Tensor&, const std::vector<size_t>&, Tensor&, std::string& error) const;
    bool convertDtype(const Tensor& input, TensorDType dtype, Tensor& out, std::string& error) const;
};

namespace ops {
bool add(const Tensor& a, const Tensor& b, Tensor& out, std::string& error);
bool sub(const Tensor& a, const Tensor& b, Tensor& out, std::string& error);
bool mul(const Tensor& a, const Tensor& b, Tensor& out, std::string& error);
bool div(const Tensor& a, const Tensor& b, Tensor& out, std::string& error);
}

} // namespace localimage::tensor
