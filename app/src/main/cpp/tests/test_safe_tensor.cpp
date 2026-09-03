#include "../safetensors/safe_tensor_file.h"
#include <cassert>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

using localimage::safetensors::SafeTensorFile;
using localimage::safetensors::TensorView;

static void writeLe64(std::ofstream& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xff));
}

static std::string makeFile(const std::string& json, const std::string& bytes) {
    char path[] = "/tmp/localimage-test-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); close(fd);
    std::ofstream out(path, std::ios::binary);
    writeLe64(out, json.size()); out.write(json.data(), static_cast<std::streamsize>(json.size()));
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); out.close();
    return path;
}

static bool load(const std::string& path, SafeTensorFile& f, std::string& err) {
    int fd = ::open(path.c_str(), O_RDONLY); assert(fd >= 0);
    return f.open(fd, err) && f.validate(err);
}

static void expectInvalid(const std::string& json, const std::string& bytes) {
    const auto path = makeFile(json, bytes); SafeTensorFile f; std::string err;
    assert(!load(path, f, err)); assert(!err.empty()); unlink(path.c_str());
}

int main() {
    const std::string json = R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[2],"data_offsets":[16,20]},"u":{"dtype":"BF16","shape":[2],"data_offsets":[20,24]}})";
    std::string bytes(24, '\0'); for (int i = 0; i < 24; ++i) bytes[i] = static_cast<char>(i);
    const auto path = makeFile(json, bytes);
    SafeTensorFile f; std::string err;
    assert(load(path, f, err));
    assert(f.tensors().size() == 3);
    const auto* a = f.findTensor("a"); assert(a && a->byte_size == 16 && a->dtype == localimage::safetensors::DType::F32);
    const auto* b = f.findTensor("b"); assert(b && b->byte_size == 4 && b->dtype == localimage::safetensors::DType::F16);
    const auto* u = f.findTensor("u"); assert(u && u->dtype == localimage::safetensors::DType::BF16);
    TensorView view; assert(f.getTensorView("a", view, err));
    assert(static_cast<const uint8_t*>(view.data())[0] == 0 && static_cast<const uint8_t*>(view.data())[15] == 15);
    f.close();
    // TensorView retains the mapping owner, so using a view after SafeTensorFile::close is safe.
    assert(static_cast<const uint8_t*>(view.data())[0] == 0 && view.byteSize() == 16);
    unlink(path.c_str());

    // Header-size overflow / impossible header.
    const auto badHeader = makeFile("{}", "");
    int fd = ::open(badHeader.c_str(), O_RDWR); assert(fd >= 0);
    const uint64_t huge = UINT64_MAX; pwrite(fd, &huge, 8, 0); close(fd);
    SafeTensorFile bad; assert(!load(badHeader, bad, err)); unlink(badHeader.c_str());

    // Invalid offsets: begin > end.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", std::string(4, '\0'));
    // Invalid offsets: beyond data region.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})", std::string(4, '\0'));
    // Truncated tensor: F32[2] requires 8 bytes, only 4 are declared.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", std::string(4, '\0'));
    // Shape multiplication overflow.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}})", "");
    // Invalid JSON.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]})", std::string(4, '\0'));
    // Duplicate JSON object key must be rejected.
    expectInvalid(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", std::string(4, '\0'));

    // mmap failure: invalid descriptor.
    localimage::safetensors::MappedFile mapped; assert(!mapped.map(-1, 0, 1, err));

    std::cout << "All SafeTensors/mmap tests passed\n";
}
