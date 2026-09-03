#include "runtime_cache.h"

#include <cerrno>
#include <cstring>
#include <filesystem>

namespace localimage::runtime {
namespace {
std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + '/' + b;
}
}

RuntimeCache::RuntimeCache(std::string root_directory) : root_directory_(std::move(root_directory)) {}

std::string RuntimeCache::directoryFor(const std::string& cache_key) const {
    return joinPath(root_directory_, cache_key);
}
std::string RuntimeCache::graphPath(const std::string& cache_key) const { return joinPath(directoryFor(cache_key), "graph.cache"); }
std::string RuntimeCache::shaderPath(const std::string& cache_key) const { return joinPath(directoryFor(cache_key), "shader.cache"); }
std::string RuntimeCache::pipelinePath(const std::string& cache_key) const { return joinPath(directoryFor(cache_key), "pipeline.cache"); }

bool RuntimeCache::isCompatible(const CacheKeyInput& expected, const std::string& stored_cache_key) const {
    const std::string current = CacheKey::build(expected);
    return !current.empty() && current == stored_cache_key;
}

bool RuntimeCache::invalidate(const std::string& cache_key, std::string& error) const {
    if (cache_key.empty()) { error = "empty cache key"; return false; }
    std::error_code ec;
    std::filesystem::remove_all(directoryFor(cache_key), ec);
    if (ec) {
        error = "cache invalidation failed: " + ec.message();
        return false;
    }
    return true;
}

} // namespace localimage::runtime
