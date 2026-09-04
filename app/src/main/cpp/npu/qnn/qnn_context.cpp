#include "qnn_context.h"

#include <sstream>
#include <cstring>
#include <cstdlib>
#include <sys/system_properties.h>

#ifdef __ANDROID__
#include <dlfcn.h>
#endif

namespace localimage {
namespace npu {
namespace qnn {

// ============================================================================
// SoC / DSP version detection
// ============================================================================

static DspVersion detectDspFromSoc(const std::string& socModel) {
    // Qualcomm SoC model names and their Hexagon DSP versions
    // Snapdragon 8 Gen3 = SM8650 -> v75
    // Snapdragon 8 Elite = SM8750 -> v79
    // Snapdragon 8 Elite Gen5 = SM8850 -> v81
    // Snapdragon 8 Gen2 = SM8550 -> v73

    if (socModel.find("SM8550") != std::string::npos ||
        socModel.find("8 Gen2") != std::string::npos) {
        return DspVersion::V73;
    }
    if (socModel.find("SM8650") != std::string::npos ||
        socModel.find("8 Gen3") != std::string::npos ||
        socModel.find("8gen3") != std::string::npos) {
        return DspVersion::V75;
    }
    if (socModel.find("SM8750") != std::string::npos ||
        socModel.find("8 Elite") != std::string::npos ||
        socModel.find("8elite") != std::string::npos) {
        return DspVersion::V79;
    }
    if (socModel.find("SM8850") != std::string::npos ||
        socModel.find("8 Elite Gen5") != std::string::npos ||
        socModel.find("8 Elite 2") != std::string::npos) {
        return DspVersion::V81;
    }
    return DspVersion::Unknown;
}

static std::string readSystemProperty(const char* key) {
    char value[PROP_VALUE_MAX] = {0};
    int len = __system_property_get(key, value);
    if (len > 0) return std::string(value, len);
    return "";
}

// ============================================================================
// QnnContext implementation
// ============================================================================

QnnContext::QnnContext() = default;

QnnContext::~QnnContext() {
    shutdown();
}

bool QnnContext::initialize(std::string& error) {
    if (initialized_) {
        error = "QNN context already initialized";
        return available_;
    }
    initialized_ = true;

    // Step 1: Detect SoC / DSP version
    if (!detectSocVersion(error)) {
        caps_.errorMessage = error;
        return false;
    }

    // Step 2: Check minimum version requirement (v75 = 8 Gen3)
    if (static_cast<int>(caps_.dspVersion) < static_cast<int>(kMinSupportedDspVersion)) {
        std::ostringstream os;
        os << "DSP version " << dspVersionName(caps_.dspVersion)
           << " is below minimum required " << dspVersionName(kMinSupportedDspVersion);
        error = os.str();
        caps_.errorMessage = error;
        return false;
    }

#ifdef LOCALIMAGE_QNN
    // Step 3: Load backend library
    if (!loadBackendLibrary(error)) {
        caps_.errorMessage = error;
        return false;
    }

    // Step 4: Create device
    if (!createDevice(error)) {
        caps_.errorMessage = error;
        return false;
    }

    // Step 5: Create context
    if (!createContext(error)) {
        caps_.errorMessage = error;
        return false;
    }

    // Step 6: Probe supported ops
    if (!probeSupportedOps(error)) {
        // Non-fatal: continue with default op list
    }
#else
    // Without QNN SDK linked, report unavailable but preserve detection info
    error = "QNN SDK not compiled in (LOCALIMAGE_QNN not defined)";
    caps_.errorMessage = error;
    caps_.available = false;
    return false;
#endif

    caps_.available = true;
    available_ = true;
    return true;
}

bool QnnContext::detectSocVersion(std::string& error) {
    // Try multiple sources for SoC info
    std::string socModel = readSystemProperty("ro.soc.model");
    if (socModel.empty()) {
        socModel = readSystemProperty("ro.board.platform");
    }
    if (socModel.empty()) {
        socModel = readSystemProperty("ro.product.board");
    }
    if (socModel.empty()) {
        // Try hardware name
        socModel = readSystemProperty("ro.hardware");
    }

    caps_.socName = socModel;
    caps_.dspVersion = detectDspFromSoc(socModel);

    if (caps_.dspVersion == DspVersion::Unknown) {
        error = "Unable to detect Qualcomm SoC / DSP version. SoC: " + socModel;
        return false;
    }

    caps_.deviceName = "Qualcomm Hexagon DSP " + dspVersionName(caps_.dspVersion);
    return true;
}

#ifdef LOCALIMAGE_QNN
// ---- QNN SDK-dependent implementations ----

bool QnnContext::loadBackendLibrary(std::string& error) {
    const std::string skelLib = selectSkelLibrary();

    // Load skel library first (HTP v75/v79/v81 skel)
    // The skel library implements the DSP-side code and must be loaded
    // before the backend so the backend can find the skel symbols
    if (!skelLib.empty()) {
        skelLibHandle_ = dlopen(skelLib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!skelLibHandle_) {
            error = std::string("Failed to load HTP skel library: ") + dlerror();
            return false;
        }
    }

    // Load HTP backend library
    backendLibHandle_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (!backendLibHandle_) {
        // Fallback: try system library path
        backendLibHandle_ = dlopen("/vendor/lib64/libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!backendLibHandle_) {
        error = std::string("Failed to load QNN HTP backend: ") + dlerror();
        return false;
    }

    // Resolve QnnInterface_getProviders entry point
    // Standard QNN signature:
    //   Qnn_ErrorHandle_t QnnInterface_getProviders(
    //       uint32_t apiVersion,
    //       const QnnInterface_t** providers,
    //       uint32_t* numProviders);
    using GetQnnInterface_fn = Qnn_ErrorHandle_t (*)(
        uint32_t, const QnnInterface_t**, uint32_t*);
    auto getInterface = reinterpret_cast<GetQnnInterface_fn>(
        dlsym(backendLibHandle_, "QnnInterface_getProviders"));
    if (!getInterface) {
        error = "Failed to find QnnInterface_getProviders in backend library: " +
                std::string(dlerror());
        return false;
    }

    // Query available interface providers
    const QnnInterface_t* providers = nullptr;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t status = getInterface(
        QNN_API_VERSION, &providers, &numProviders);
    if (status != QNN_SUCCESS) {
        error = "QnnInterface_getProviders failed. Error code: " +
                std::to_string(status);
        return false;
    }
    if (!providers || numProviders == 0) {
        error = "No QNN interface providers returned by backend";
        return false;
    }

    // Select the first provider (HTP backend typically returns one provider)
    // In multi-backend scenarios we would iterate to find the matching backend type
    qnnInterface_ = const_cast<QnnInterface_t*>(providers);

    // Validate core function tables are present
    if (!qnnInterface_->device) {
        error = "QNN interface missing device function table";
        return false;
    }
    if (!qnnInterface_->context) {
        error = "QNN interface missing context function table";
        return false;
    }
    if (!qnnInterface_->graph) {
        error = "QNN interface missing graph function table";
        return false;
    }

    return true;
}

bool QnnContext::createDevice(std::string& error) {
    if (!qnnInterface_ || !qnnInterface_->device) {
        error = "QNN device interface not available";
        return false;
    }

    // Configure HTP device options
    // QNN device configs are a tagged array: each config entry has an option type
    // and a corresponding value. The array is terminated with QNN_DEVICE_CONFIG_OPTION_END.
    //
    // Key HTP device configurations:
    // - Performance mode: controls DVFS and power budget
    // - RPC polling: reduces latency for small graphs by busy-waiting
    // - Device ID: selects which HTP core to use (for multi-HTP SoCs)
    std::vector<QnnDevice_Config_t> deviceConfigs;

    // Performance mode: high_performance for inference workloads
    {
        QnnDevice_Config_t cfg;
        cfg.option = QNN_DEVICE_CONFIG_OPTION_PERF_MODE;
        cfg.perfMode = QNN_PERF_MODE_BURST; // max performance, higher power
        deviceConfigs.push_back(cfg);
    }

    // RPC polling: reduces submit/completion latency for small graphs
    {
        QnnDevice_Config_t cfg;
        cfg.option = QNN_DEVICE_CONFIG_OPTION_RPC_POLLING;
        cfg.rpcPolling = 1; // enable
        deviceConfigs.push_back(cfg);
    }

    // Device ID: use default HTP core (device 0)
    {
        QnnDevice_Config_t cfg;
        cfg.option = QNN_DEVICE_CONFIG_OPTION_DEVICE_ID;
        cfg.deviceId = 0;
        deviceConfigs.push_back(cfg);
    }

    // Terminator config entry
    {
        QnnDevice_Config_t cfg;
        cfg.option = QNN_DEVICE_CONFIG_OPTION_END;
        deviceConfigs.push_back(cfg);
    }

    Qnn_ErrorHandle_t status = qnnInterface_->device->create(
        deviceConfigs.data(),
        static_cast<uint32_t>(deviceConfigs.size()),
        &deviceHandle_);

    if (status != QNN_SUCCESS) {
        error = "Failed to create QNN device. Error code: " + std::to_string(status);
        return false;
    }

    return true;
}

bool QnnContext::createContext(std::string& error) {
    if (!qnnInterface_ || !qnnInterface_->context) {
        error = "QNN context interface not available";
        return false;
    }

    // Configure QNN context
    // Context configs follow the same tagged-array pattern as device configs.
    // Key context configurations:
    // - Priority: scheduling priority among multiple contexts
    // - Memory configuration: heap size, DMA buffer pool size
    std::vector<QnnContext_Config_t> contextConfigs;

    // Context priority: normal (can be adjusted for multi-context scenarios)
    {
        QnnContext_Config_t cfg;
        cfg.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
        cfg.priority = QNN_CONTEXT_PRIORITY_NORMAL;
        contextConfigs.push_back(cfg);
    }

    // Terminator
    {
        QnnContext_Config_t cfg;
        cfg.option = QNN_CONTEXT_CONFIG_OPTION_END;
        contextConfigs.push_back(cfg);
    }

    Qnn_ErrorHandle_t status = qnnInterface_->context->create(
        deviceHandle_,
        contextConfigs.data(),
        static_cast<uint32_t>(contextConfigs.size()),
        &contextHandle_);

    if (status != QNN_SUCCESS) {
        error = "Failed to create QNN context. Error code: " + std::to_string(status);
        return false;
    }

    return true;
}

bool QnnContext::probeSupportedOps(std::string& error) {
    // Query backend for supported operations via tensor/operation query API.
    // In a full implementation, we would iterate through all QNN op types
    // and call the appropriate query function to check runtime support.
    // For now, we report the standard HTP FP16 supported ops list.
    //
    // The QNN SDK provides:
    //   graph->operationInfo() or backend->getOpInfo()
    // for querying supported operations on the current device.
    caps_.supportedOps = {
        QnnOpType::Add,
        QnnOpType::Sub,
        QnnOpType::Mul,
        QnnOpType::Div,
        QnnOpType::MatMul,
        QnnOpType::BatchMatMul,
        QnnOpType::Conv2D,
        QnnOpType::DepthwiseConv2D,
        QnnOpType::Softmax,
        QnnOpType::LayerNorm,
        QnnOpType::GroupNorm,
        QnnOpType::SiLU,
        QnnOpType::GELU,
        QnnOpType::ReLU,
        QnnOpType::Exp,
        QnnOpType::Sqrt,
        QnnOpType::Rsqrt,
        QnnOpType::Clamp,
        QnnOpType::Reshape,
        QnnOpType::Transpose,
        QnnOpType::Slice,
        QnnOpType::Concat,
        QnnOpType::Broadcast,
        QnnOpType::UpsampleNearest,
        QnnOpType::UpsampleBilinear,
        QnnOpType::ReduceMean,
        QnnOpType::ReduceSum,
        QnnOpType::Sigmoid,
        QnnOpType::Tanh,
        QnnOpType::Pad,
        QnnOpType::StridedSlice,
        QnnOpType::Squeeze,
        QnnOpType::Unsqueeze,
    };
    return true;
}

#else
// ---- Stub implementations when QNN SDK is not available ----

bool QnnContext::loadBackendLibrary(std::string& error) {
    error = "QNN SDK not available";
    return false;
}

bool QnnContext::createDevice(std::string& error) {
    error = "QNN SDK not available";
    return false;
}

bool QnnContext::createContext(std::string& error) {
    error = "QNN SDK not available";
    return false;
}

bool QnnContext::probeSupportedOps(std::string& error) {
    error = "QNN SDK not available";
    return false;
}

#endif // LOCALIMAGE_QNN

std::string QnnContext::selectSkelLibrary() const {
    // Select the appropriate skel library based on DSP version
    switch(caps_.dspVersion) {
        case DspVersion::V75: return "libQnnHtpV75Skel.so";
        case DspVersion::V79: return "libQnnHtpV79Skel.so";
        case DspVersion::V81: return "libQnnHtpV81Skel.so";
        case DspVersion::V73: return "libQnnHtpV73Skel.so";
        default: return "";
    }
}

void* QnnContext::allocateDeviceMemory(size_t bytes, std::string& error) {
    if (!available_) {
        error = "QNN context not available for device memory allocation";
        return nullptr;
    }
    // In a full implementation, this would use QNN memory management API
    // or ION buffer allocation. For now, use posix_memalign as fallback.
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 128, bytes) != 0) {
        error = "Failed to allocate device memory";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(memMutex_);
    deviceAllocations_[ptr] = bytes;
    return ptr;
}

void QnnContext::freeDeviceMemory(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(memMutex_);
    auto it = deviceAllocations_.find(ptr);
    if (it != deviceAllocations_.end()) {
        free(ptr);
        deviceAllocations_.erase(it);
    }
}

void* QnnContext::allocateSharedMemory(size_t bytes, std::string& error) {
    if (!available_) {
        error = "QNN context not available for shared memory allocation";
        return nullptr;
    }
    // ION / dmabuf allocation would go here for zero-copy
    // For now, use aligned allocation as CPU-side fallback
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, bytes) != 0) {
        error = "Failed to allocate shared memory";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(memMutex_);
    sharedAllocations_[ptr] = bytes;
    return ptr;
}

void QnnContext::freeSharedMemory(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(memMutex_);
    auto it = sharedAllocations_.find(ptr);
    if (it != sharedAllocations_.end()) {
        free(ptr);
        sharedAllocations_.erase(it);
    }
}

void QnnContext::shutdown() {
    if (!initialized_) return;

    // Free all tracked memory
    {
        std::lock_guard<std::mutex> lock(memMutex_);
        for (auto& [ptr, size] : deviceAllocations_) {
            free(ptr);
        }
        deviceAllocations_.clear();
        for (auto& [ptr, size] : sharedAllocations_) {
            free(ptr);
        }
        sharedAllocations_.clear();
    }

#ifdef LOCALIMAGE_QNN
    if (qnnInterface_ && qnnInterface_->context && contextHandle_) {
        qnnInterface_->context->free(contextHandle_);
        contextHandle_ = nullptr;
    }
    if (qnnInterface_ && qnnInterface_->device && deviceHandle_) {
        qnnInterface_->device->free(deviceHandle_);
        deviceHandle_ = nullptr;
    }
    if (skelLibHandle_) {
        dlclose(skelLibHandle_);
        skelLibHandle_ = nullptr;
    }
    if (backendLibHandle_) {
        dlclose(backendLibHandle_);
        backendLibHandle_ = nullptr;
    }
    qnnInterface_ = nullptr;
#endif

    available_ = false;
    initialized_ = false;
}

// ============================================================================
// Global singleton
// ============================================================================

QnnContext& getQnnContext() {
    static QnnContext instance;
    return instance;
}

} // namespace qnn
} // namespace npu
} // namespace localimage
