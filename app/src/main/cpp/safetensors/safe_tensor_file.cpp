#include "safe_tensor_file.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace localimage::safetensors {

const char* dtypeName(DType dtype) {
    switch (dtype) {
        case DType::F16: return "F16";
        case DType::F32: return "F32";
        case DType::BF16: return "BF16";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
        case DType::F8_E4M3: return "F8_E4M3";
        case DType::F8_E5M2: return "F8_E5M2";
        default: return "UNKNOWN";
    }
}

uint64_t dtypeSize(DType dtype) {
    switch (dtype) {
        case DType::F16: case DType::F32: case DType::BF16: return dtype == DType::F32 ? 4 : 2;
        case DType::I8: case DType::U8: case DType::F8_E4M3: case DType::F8_E5M2: return 1;
        default: return 0;
    }
}

bool parseDType(const std::string& text, DType& dtype) {
    if (text == "F16") { dtype = DType::F16; return true; }
    if (text == "F32") { dtype = DType::F32; return true; }
    if (text == "BF16") { dtype = DType::BF16; return true; }
    if (text == "I8") { dtype = DType::I8; return true; }
    if (text == "U8") { dtype = DType::U8; return true; }
    if (text == "F8_E4M3") { dtype = DType::F8_E4M3; return true; }
    if (text == "F8_E5M2") { dtype = DType::F8_E5M2; return true; }
    dtype = DType::Unknown;
    return false;
}

MappedFile::~MappedFile() { unmap(); }

bool MappedFile::map(int fd, uint64_t offset, uint64_t length, std::string& error) {
    return mapRange(fd, offset, length, error);
}

bool MappedFile::mapRange(int fd, uint64_t offset, uint64_t length, std::string& error) {
    unmap();
    if (fd < 0) { error = "invalid file descriptor"; return false; }
    if (length == 0) { error = "cannot mmap zero bytes"; return false; }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { error = "failed to query page size"; return false; }
    const uint64_t page = static_cast<uint64_t>(page_size);
    const uint64_t page_offset = offset / page * page;
    const uint64_t delta = offset - page_offset;
    if (length > std::numeric_limits<uint64_t>::max() - delta) { error = "mmap length overflow"; return false; }
    const uint64_t map_length_u64 = delta + length;
    if (map_length_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) { error = "mmap length exceeds size_t"; return false; }
    if (page_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) { error = "mmap offset exceeds off_t"; return false; }

    void* base = ::mmap(nullptr, static_cast<size_t>(map_length_u64), PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(page_offset));
    if (base == MAP_FAILED) { error = std::string("mmap failed: ") + std::strerror(errno); return false; }
    mapped_base_ = base;
    mapped_length_ = static_cast<size_t>(map_length_u64);
    requested_length_ = length;
    delta_ = delta;
    return true;
}

const uint8_t* MappedFile::data() const {
    if (!mapped_base_) return nullptr;
    return static_cast<const uint8_t*>(mapped_base_) + delta_;
}

void MappedFile::unmap() {
    if (mapped_base_) ::munmap(mapped_base_, mapped_length_);
    mapped_base_ = nullptr;
    mapped_length_ = 0;
    requested_length_ = 0;
    delta_ = 0;
}

SafeTensorFile::~SafeTensorFile() { close(); }

bool SafeTensorFile::open(int fd, std::string& error) {
    close();
    if (fd < 0) { error = "invalid file descriptor"; return false; }
    struct stat st{};
    if (::fstat(fd, &st) != 0) { error = std::string("fstat failed: ") + std::strerror(errno); ::close(fd); return false; }
    if (st.st_size < 0) { error = "negative file size"; ::close(fd); return false; }
    fd_ = fd;
    file_size_ = static_cast<uint64_t>(st.st_size);
    if (file_size_ < 8) { error = "file_size < 8"; close(); return false; }
    mapping_ = std::make_shared<MappedFile>();
    if (!mapping_->map(fd_, 0, file_size_, error)) { close(); return false; }
    return true;
}

bool SafeTensorFile::readU64LE(const uint8_t* p, uint64_t& value, std::string& error) const {
    if (!p) { error = "null header pointer"; return false; }
    value = static_cast<uint64_t>(p[0]) |
            (static_cast<uint64_t>(p[1]) << 8) |
            (static_cast<uint64_t>(p[2]) << 16) |
            (static_cast<uint64_t>(p[3]) << 24) |
            (static_cast<uint64_t>(p[4]) << 32) |
            (static_cast<uint64_t>(p[5]) << 40) |
            (static_cast<uint64_t>(p[6]) << 48) |
            (static_cast<uint64_t>(p[7]) << 56);
    (void)error;
    return true;
}

bool SafeTensorFile::parseHeader(std::string& error) {
    tensors_.clear(); header_json_.clear(); header_parsed_ = false;
    if (fd_ < 0 || !mapping_ || !mapping_->data()) { error = "file is not open"; return false; }
    if (file_size_ < 8) { error = "file_size < 8"; return false; }
    if (!readU64LE(mapping_->data(), header_size_, error)) return false;
    if (header_size_ > kMaxHeaderSize) { error = "header_size exceeds 64 MiB limit"; return false; }
    if (header_size_ > file_size_ - 8) { error = "header_size exceeds file payload"; return false; }
    data_start_ = 8 + header_size_;
    const uint8_t* header_ptr = mapping_->data() + 8;
    header_json_.assign(reinterpret_cast<const char*>(header_ptr), static_cast<size_t>(header_size_));

    json::Value root;
    if (!json::parse(header_json_.data(), header_json_.size(), root, error)) { error = "invalid JSON header: " + error; return false; }
    const auto* object = root.object();
    if (!object) { error = "SafeTensors header must be a JSON object"; return false; }

    for (const auto& [name, value] : *object) {
        if (name == "__metadata__") continue;
        if (!parseTensorObject(name, value, error)) { error = "tensor '" + name + "': " + error; return false; }
    }
    header_parsed_ = true;
    return true;
}

bool SafeTensorFile::checkedMul(uint64_t a, uint64_t b, uint64_t& out) const {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b; return true;
}

bool SafeTensorFile::parseTensorObject(const std::string& name, const json::Value& value, std::string& error) {
    const auto* object = value.object();
    if (!object) { error = "metadata is not an object"; return false; }
    const auto dtype_it = object->find("dtype");
    const auto shape_it = object->find("shape");
    const auto offsets_it = object->find("data_offsets");
    if (dtype_it == object->end() || shape_it == object->end() || offsets_it == object->end()) { error = "missing dtype, shape, or data_offsets"; return false; }

    std::string dtype_text;
    if (!dtype_it->second.getString(dtype_text)) { error = "dtype must be a string"; return false; }
    DType dtype = DType::Unknown;
    parseDType(dtype_text, dtype);

    const auto* shape = shape_it->second.array();
    if (!shape) { error = "shape must be an array"; return false; }
    std::vector<uint64_t> dims;
    dims.reserve(shape->size());
    uint64_t elements = 1;
    for (const auto& dim : *shape) {
        uint64_t d = 0;
        if (!dim.getUnsigned(d)) { error = "shape dimension must be a non-negative integer"; return false; }
        if (!checkedMul(elements, d, elements)) { error = "shape element count overflow"; return false; }
        dims.push_back(d);
    }

    const auto* offsets = offsets_it->second.array();
    if (!offsets || offsets->size() != 2) { error = "data_offsets must contain exactly two integers"; return false; }
    uint64_t begin = 0, end = 0;
    if (!(*offsets)[0].getUnsigned(begin) || !(*offsets)[1].getUnsigned(end)) { error = "data_offsets must be non-negative integers"; return false; }
    if (begin > end) { error = "data_offsets begin > end"; return false; }
    const uint64_t available = file_size_ - data_start_;
    if (begin > available || end > available) { error = "data_offsets exceed tensor data region"; return false; }
    const uint64_t byte_size = end - begin;

    const uint64_t element_size = dtypeSize(dtype);
    if (element_size != 0) {
        uint64_t calculated = 0;
        if (!checkedMul(elements, element_size, calculated)) { error = "tensor byte size overflow"; return false; }
        if (calculated != byte_size) { error = "data_offsets size does not match shape × dtype size"; return false; }
    }

    TensorInfo info;
    info.name = name; info.dtype = dtype; info.shape = std::move(dims);
    info.data_begin = begin; info.data_end = end; info.byte_size = byte_size;
    tensors_.emplace(name, std::move(info));
    return true;
}

bool SafeTensorFile::validate(std::string& error) {
    if (!header_parsed_ && !parseHeader(error)) return false;
    return true;
}

const TensorInfo* SafeTensorFile::findTensor(const std::string& name) const {
    auto it = tensors_.find(name); return it == tensors_.end() ? nullptr : &it->second;
}

bool SafeTensorFile::getTensorView(const std::string& name, TensorView& out, std::string& error) const {
    const TensorInfo* info = findTensor(name);
    if (!info) { error = "tensor not found"; return false; }
    if (info->dtype != DType::F16 && info->dtype != DType::F32 &&
        info->dtype != DType::BF16 && info->dtype != DType::I8 && info->dtype != DType::U8) {
        error = "Unsupported dtype: " + std::string(dtypeName(info->dtype));
        return false;
    }
    if (info->data_begin > file_size_ - data_start_ || info->data_end > file_size_ - data_start_) { error = "tensor range invalid"; return false; }
    std::vector<uint64_t> stride(info->shape.size(), 1); uint64_t s = 1; for (size_t i = info->shape.size(); i-- > 0;) { stride[i] = s; const uint64_t d = info->shape[i]; if (d != 0 && s > std::numeric_limits<uint64_t>::max() / d) { error = "stride overflow"; return false; } s *= d; }
    out = TensorView(mapping_->data() + data_start_ + info->data_begin, info->byte_size, info->dtype, info->shape, std::move(stride), mapping_);
    return true;
}

void SafeTensorFile::close() {
    // Releasing the owner is enough: outstanding TensorView instances retain the mapping.
    // MappedFile::destructor performs munmap() when the last owner is gone.
    mapping_.reset();
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; file_size_ = 0; header_size_ = 0; data_start_ = 0; header_json_.clear(); tensors_.clear(); header_parsed_ = false;
}

} // namespace localimage::safetensors
