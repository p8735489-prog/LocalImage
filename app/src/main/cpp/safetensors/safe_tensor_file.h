#pragma once

#include "json_value.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace localimage::safetensors {

enum class DType { F16, F32, BF16, I8, U8, F8_E4M3, F8_E5M2, Unknown };

const char* dtypeName(DType dtype);
uint64_t dtypeSize(DType dtype);
bool parseDType(const std::string& text, DType& dtype);

struct TensorInfo {
    std::string name;
    DType dtype = DType::Unknown;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t byte_size = 0;
};

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool map(int fd, uint64_t offset, uint64_t length, std::string& error);
    bool mapRange(int fd, uint64_t offset, uint64_t length, std::string& error);
    const uint8_t* data() const;
    uint64_t size() const { return requested_length_; }
    void unmap();

private:
    void* mapped_base_ = nullptr;
    size_t mapped_length_ = 0;
    uint64_t requested_length_ = 0;
    uint64_t delta_ = 0;
};

class TensorView {
public:
    TensorView() = default;
    TensorView(const void* data, uint64_t byte_size, DType dtype, std::vector<uint64_t> shape, std::vector<uint64_t> stride, std::shared_ptr<const MappedFile> owner)
        : data_(data), byte_size_(byte_size), dtype_(dtype), shape_(std::move(shape)), stride_(std::move(stride)), owner_(std::move(owner)) {}

    const void* data() const { return data_; }
    uint64_t byteSize() const { return byte_size_; }
    DType dtype() const { return dtype_; }
    const std::vector<uint64_t>& shape() const { return shape_; }
    const std::vector<uint64_t>& stride() const { return stride_; }
    std::shared_ptr<const MappedFile> owner() const { return owner_; }

private:
    const void* data_ = nullptr;
    uint64_t byte_size_ = 0;
    DType dtype_ = DType::Unknown;
    std::vector<uint64_t> shape_;
    std::vector<uint64_t> stride_;
    std::shared_ptr<const MappedFile> owner_;
};

class SafeTensorFile {
public:
    SafeTensorFile() = default;
    ~SafeTensorFile();
    SafeTensorFile(const SafeTensorFile&) = delete;
    SafeTensorFile& operator=(const SafeTensorFile&) = delete;

    bool open(int fd, std::string& error);
    bool validate(std::string& error);
    bool parseHeader(std::string& error);
    const TensorInfo* findTensor(const std::string& name) const;
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }
    bool getTensorView(const std::string& name, TensorView& out, std::string& error) const;
    uint64_t fileSize() const { return file_size_; }
    const uint8_t* mappedData() const { return mapping_ ? mapping_->data() : nullptr; }
    uint64_t headerSize() const { return header_size_; }
    uint64_t dataStart() const { return data_start_; }
    const std::string& headerJson() const { return header_json_; }
    void close();

private:
    static constexpr uint64_t kMaxHeaderSize = 64ULL * 1024ULL * 1024ULL;

    int fd_ = -1;
    uint64_t file_size_ = 0;
    uint64_t header_size_ = 0;
    uint64_t data_start_ = 0;
    std::shared_ptr<MappedFile> mapping_;
    std::string header_json_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    bool header_parsed_ = false;

    bool readU64LE(const uint8_t* p, uint64_t& value, std::string& error) const;
    bool parseTensorObject(const std::string& name, const json::Value& value, std::string& error);
    bool checkedMul(uint64_t a, uint64_t b, uint64_t& out) const;
};

} // namespace localimage::safetensors
