#include "model_hash.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

namespace localimage::runtime {
namespace {

class Sha256 final {
public:
    Sha256() { reset(); }

    void update(const uint8_t* data, size_t length) {
        if (length == 0) return;
        size_t offset = 0;
        while (offset < length) {
            const size_t take = std::min(length - offset, buffer_.size() - buffer_size_);
            std::memcpy(buffer_.data() + buffer_size_, data + offset, take);
            buffer_size_ += take;
            offset += take;
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        const uint64_t original_bits = bit_count_ + static_cast<uint64_t>(buffer_size_) * 8ULL;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.end(), 0);
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            buffer_[63 - i] = static_cast<uint8_t>((original_bits >> (i * 8)) & 0xffU);
        }
        transform(buffer_.data());

        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < state_.size(); ++i) {
            out[i * 4] = static_cast<uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
        }
        return out;
    }

private:
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static uint32_t bs0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static uint32_t bs1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static uint32_t ss0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static uint32_t ss1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void reset() {
        state_ = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        buffer_.fill(0);
        buffer_size_ = 0;
        bit_count_ = 0;
    }

    void transform(const uint8_t* block) {
        uint32_t w[64]{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) w[i] = ss1(w[i - 2]) + w[i - 7] + ss0(w[i - 15]) + w[i - 16];

        uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t t1 = h + bs1(e) + ch(e,f,g) + k[i] + w[i];
            const uint32_t t2 = bs0(a) + maj(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> buffer_{};
    size_t buffer_size_ = 0;
    uint64_t bit_count_ = 0;
};

} // namespace

bool ModelHash::compute(const uint8_t* data, uint64_t size, std::string& error) {
    digest_.clear();
    if (size != 0 && data == nullptr) { error = "cannot hash a null buffer"; return false; }
    if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) { error = "model size exceeds size_t"; return false; }
    Sha256 sha;
    constexpr size_t kChunk = 1024 * 1024;
    uint64_t offset = 0;
    while (offset < size) {
        const uint64_t remaining = size - offset;
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, kChunk));
        sha.update(data + static_cast<size_t>(offset), chunk);
        offset += chunk;
    }
    const auto digest = sha.final();
    digest_.assign(digest.begin(), digest.end());
    return true;
}

std::string ModelHash::hex() const {
    if (!valid()) return {};
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest_) out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

} // namespace localimage::runtime
