#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace localimage::runtime {

class ModelHash {
public:
    static constexpr size_t kDigestSize = 32;

    ModelHash() = default;

    bool compute(const uint8_t* data, uint64_t size, std::string& error);
    const std::vector<uint8_t>& digest() const { return digest_; }
    std::string hex() const;
    bool valid() const { return digest_.size() == kDigestSize; }

private:
    std::vector<uint8_t> digest_;
};

} // namespace localimage::runtime
