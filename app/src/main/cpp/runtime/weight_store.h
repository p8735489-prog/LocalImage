#pragma once

#include "../safetensors/safe_tensor_file.h"
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

namespace localimage::runtime {

class WeightStore final {
public:
    bool attach(const safetensors::SafeTensorFile& file, std::string& error);
    bool contains(const std::string& name) const;
    bool view(const std::string& name, safetensors::TensorView& out, std::string& error) const;
    bool acquire(const std::string& name, safetensors::TensorView& out, std::string& error);
    bool release(const std::string& name, std::string& error);
    void clear();
    size_t size() const { return entries_.size(); }
    uint64_t totalBytes() const { return total_bytes_; }
    size_t residentCount() const { return resident_.size(); }
    void setResidentLimit(size_t limit) { resident_limit_ = limit; evict(); }
private:
    struct Entry { const safetensors::TensorInfo* info=nullptr; uint32_t refs=0; };
    const safetensors::SafeTensorFile* file_=nullptr;
    std::unordered_map<std::string,Entry> entries_;
    std::list<std::string> lru_;
    std::unordered_map<std::string,std::list<std::string>::iterator> resident_;
    uint64_t total_bytes_=0;
    size_t resident_limit_=32;
    void touch(const std::string& name);
    void evict();
};

} // namespace localimage::runtime
