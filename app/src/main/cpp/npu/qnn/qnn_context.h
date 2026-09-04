#pragma once

#include "qnn_types.h"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

// QNN SDK forward declarations - these types are opaque when SDK is not linked
// The actual definitions come from QNN headers when LOCALIMAGE_QNN is defined
#ifdef LOCALIMAGE_QNN
#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnDevice.h"
#endif

namespace localimage {
namespace npu {
namespace qnn {

// QNN Context manages device, backend, and memory
// This is the central hub for all NPU operations
class QnnContext {
public:
    QnnContext();
    ~QnnContext();

    QnnContext(const QnnContext&) = delete;
    QnnContext& operator=(const QnnContext&) = delete;

    // Initialize QNN with HTP backend
    // Returns true if NPU is available and initialized
    bool initialize(std::string& error);

    // Check if context is initialized and ready
    bool isAvailable() const { return available_; }

    // Get capabilities
    const QnnCapabilities& capabilities() const { return caps_; }

    // Get detected DSP version
    DspVersion dspVersion() const { return caps_.dspVersion; }

    // Device memory management
    void* allocateDeviceMemory(size_t bytes, std::string& error);
    void freeDeviceMemory(void* ptr);

    // ION / shared memory allocation (zero-copy between CPU and NPU)
    void* allocateSharedMemory(size_t bytes, std::string& error);
    void freeSharedMemory(void* ptr);

    // Get native handles (for advanced use)
#ifdef LOCALIMAGE_QNN
    QnnBackend_Handle_t backendHandle() const { return backendHandle_; }
    QnnDevice_Handle_t deviceHandle() const { return deviceHandle_; }
    QnnContext_Handle_t contextHandle() const { return contextHandle_; }
#endif

private:
    // Detect SoC and DSP version from system
    bool detectSocVersion(std::string& error);

    // Load QNN backend library (libQnnHtp.so)
    bool loadBackendLibrary(std::string& error);

    // Create QNN device
    bool createDevice(std::string& error);

    // Create QNN context
    bool createContext(std::string& error);

    // Probe supported operations
    bool probeSupportedOps(std::string& error);

    // Cleanup all resources
    void shutdown();

    bool available_ = false;
    bool initialized_ = false;
    QnnCapabilities caps_;

    // Memory tracking
    std::mutex memMutex_;
    std::unordered_map<void*, size_t> deviceAllocations_;
    std::unordered_map<void*, size_t> sharedAllocations_;

#ifdef LOCALIMAGE_QNN
    // QNN native handles
    QnnBackend_Handle_t backendHandle_ = nullptr;
    QnnDevice_Handle_t deviceHandle_ = nullptr;
    QnnContext_Handle_t contextHandle_ = nullptr;
    QnnInterface_t* qnnInterface_ = nullptr;

    // Library handles
    void* backendLibHandle_ = nullptr;
    void* skelLibHandle_ = nullptr;
#endif

    // HTP skel library path selection
    std::string selectSkelLibrary() const;
};

// Global singleton accessor
QnnContext& getQnnContext();

} // namespace qnn
} // namespace npu
} // namespace localimage
