#pragma once

#include "cache_key.h"

#include <string>

namespace localimage::runtime {

class RuntimeCache {
public:
    explicit RuntimeCache(std::string root_directory);

    std::string directoryFor(const std::string& cache_key) const;
    std::string graphPath(const std::string& cache_key) const;
    std::string shaderPath(const std::string& cache_key) const;
    std::string pipelinePath(const std::string& cache_key) const;

    // Foundation only: this class does not compile shaders or create pipelines yet.
    bool isCompatible(const CacheKeyInput& expected, const std::string& stored_cache_key) const;
    bool invalidate(const std::string& cache_key, std::string& error) const;

private:
    std::string root_directory_;
};

} // namespace localimage::runtime
