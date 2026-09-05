#include <jni.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <mutex>
#include <unordered_set>
#include <cmath>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>

#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_compute.h"
#include "safetensors/safe_tensor_file.h"
#include "runtime/model_hash.h"
#include "runtime/cache_key.h"
#include "runtime/device_info.h"
#include "tensor/tensor.h"
#include "models/model_detector.h"
#include "runtime/resolution_policy.h"
#include "npu/npu_backend.h"

using localimage::safetensors::SafeTensorFile;
using localimage::safetensors::TensorInfo;
using localimage::safetensors::TensorView;
using localimage::safetensors::dtypeName;
using localimage::runtime::ModelHash;
using localimage::runtime::CacheKey;
using localimage::runtime::CacheKeyInput;
using localimage::tensor::Tensor;
using localimage::tensor::TensorDType;
using localimage::tensor::TensorDevice;
using localimage::tensor::TensorRuntime;
using localimage::tensor::TensorShape;
using localimage::tensor::ops::add;
using localimage::tensor::ops::sub;
using localimage::tensor::ops::mul;
using localimage::tensor::ops::div;

namespace {

std::mutex g_handle_mutex;
std::unordered_set<SafeTensorFile*> g_handles;

SafeTensorFile* fromHandle(jlong handle) {
    if (handle == 0) return nullptr;
    auto* ptr = reinterpret_cast<SafeTensorFile*>(static_cast<uintptr_t>(handle));
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    return g_handles.find(ptr) == g_handles.end() ? nullptr : ptr;
}

bool registerHandle(SafeTensorFile* ptr) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    return g_handles.insert(ptr).second;
}

SafeTensorFile* unregisterHandle(jlong handle) {
    if (handle == 0) return nullptr;
    auto* ptr = reinterpret_cast<SafeTensorFile*>(static_cast<uintptr_t>(handle));
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    const auto it = g_handles.find(ptr);
    if (it == g_handles.end()) return nullptr;
    g_handles.erase(it);
    return ptr;
}

void throwJava(JNIEnv* env, const char* cls, const std::string& message) {
    jclass c = env->FindClass(cls);
    if (c) env->ThrowNew(c, message.c_str());
}

std::string shapeString(const std::vector<uint64_t>& shape) {
    std::ostringstream os; os << '[';
    for (size_t i = 0; i < shape.size(); ++i) { if (i) os << ", "; os << shape[i]; }
    os << ']'; return os.str();
}

std::string tensorSummary(const TensorInfo& t) {
    std::ostringstream os;
    os << t.name << "\n"
       << "dtype: " << dtypeName(t.dtype) << "\n"
       << "shape: " << shapeString(t.shape) << "\n"
       << "offset: [" << t.data_begin << ", " << t.data_end << ")\n"
       << "size: " << t.byte_size << " bytes";
    return os.str();
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_initializeVulkan(JNIEnv* env, jclass) {
    localimage::VulkanContext context;
    std::string error;
    if (!context.initialize(error)) {
        const std::string message = "Vulkan 初始化失败\n" + error;
        return env->NewStringUTF(message.c_str());
    }
    const std::string message = "Vulkan 初始化成功\n" + context.deviceSummary();
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeOpenSafeTensor(JNIEnv* env, jclass, jint fd) {
    if (fd < 0) { throwJava(env, "java/lang/IllegalArgumentException", "invalid file descriptor"); return 0; }
    auto* file = new SafeTensorFile();
    std::string error;
    if (!file->open(fd, error) || !file->validate(error)) {
        delete file;
        throwJava(env, "java/io/IOException", error);
        return 0;
    }
    if (!registerHandle(file)) {
        delete file;
        throwJava(env, "java/io/IllegalStateException", "failed to register SafeTensor handle");
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(file));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetFileSize(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return 0; }
    return static_cast<jlong>(file->fileSize());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetTensorCount(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return 0; }
    return static_cast<jlong>(file->tensors().size());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetTensorInfo(JNIEnv* env, jclass, jlong handle, jint index) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    if (index < 0 || static_cast<size_t>(index) >= file->tensors().size()) { throwJava(env, "java/lang/IndexOutOfBoundsException", "tensor index out of range"); return nullptr; }
    std::vector<const TensorInfo*> infos;
    infos.reserve(file->tensors().size());
    for (const auto& [name, info] : file->tensors()) infos.push_back(&info);
    std::sort(infos.begin(), infos.end(), [](const TensorInfo* a, const TensorInfo* b) { return a->name < b->name; });
    return env->NewStringUTF(tensorSummary(*infos[static_cast<size_t>(index)]).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetFirstSupportedTensorBytes(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    std::vector<const TensorInfo*> infos;
    for (const auto& [name, info] : file->tensors()) infos.push_back(&info);
    std::sort(infos.begin(), infos.end(), [](const TensorInfo* a, const TensorInfo* b) { return a->name < b->name; });
    for (const auto* info : infos) {
        if (info->dtype != localimage::safetensors::DType::F16 && info->dtype != localimage::safetensors::DType::F32) continue;
        TensorView view; std::string error;
        if (!file->getTensorView(info->name, view, error)) continue;
        const auto* bytes = static_cast<const unsigned char*>(view.data());
        const size_t n = std::min<uint64_t>(16, view.byteSize());
        std::ostringstream os;
        os << info->name << ": ";
        for (size_t i = 0; i < n; ++i) os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << (i + 1 == n ? "" : " ");
        return env->NewStringUTF(os.str().c_str());
    }
    return env->NewStringUTF("No F16/F32 tensor available");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetGenerationReadiness(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) {
        throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle");
        return nullptr;
    }
    std::string validation;
    if (!file->validate(validation)) {
        throwJava(env, "java/io/IOException", validation);
        return nullptr;
    }

    const auto detection = localimage::models::ModelDetector{}.detect(*file);
    std::ostringstream os;
    os << "architecture=" << localimage::models::architectureName(detection.architecture) << "\n";
    os << "confidence=" << std::fixed << std::setprecision(2) << detection.confidence << "\n";
    os << "components: UNet=" << (detection.components.unet ? "yes" : "no")
       << ", VAE=" << (detection.components.vae ? "yes" : "no")
       << ", CLIP=" << (detection.components.clip ? "yes" : "no")
       << ", OpenCLIP=" << (detection.components.openclip ? "yes" : "no")
       << ", T5=" << (detection.components.t5 ? "yes" : "no")
       << ", Transformer=" << (detection.components.transformer ? "yes" : "no") << "\n";

    const bool complete =
        detection.architecture == localimage::models::Architecture::StableDiffusion15 &&
        detection.components.unet && detection.components.vae && detection.components.clip;

    // This is deliberately a capability gate rather than an image generator.
    // The current tree has a real tensor/graph/Vulkan foundation, but it does
    // not yet contain the complete SD1.x text-encoder -> UNet -> VAE pipeline.
    os << "generation=" << (complete ? "pipeline-not-yet-wired" : "blocked") << "\n";
    if (detection.architecture == localimage::models::Architecture::Unknown) {
        os << "reason=无法可靠识别模型架构，禁止进入生成链路";
    } else if (!complete) {
        os << "reason=" << (detection.reason.empty() ? "缺少可执行的完整模型组件" : detection.reason);
    } else {
        os << "reason=SD1.x 组件完整，但当前版本尚未把文本编码器、UNet、Scheduler、VAE 串成真实生成图";
    }
    return env->NewStringUTF(os.str().c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeValidateModel(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return JNI_FALSE; }
    std::string error;
    if (!file->validate(error)) { throwJava(env, "java/io/IOException", error); return JNI_FALSE; }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeCloseSafeTensor(JNIEnv*, jclass, jlong handle) {
    if (auto* file = unregisterHandle(handle)) delete file;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetModelArchitecture(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    const localimage::models::Detection d = localimage::models::ModelDetector{}.detect(*file);
    const auto info = localimage::runtime::ResolutionPolicy::defaults(d.architecture);
    std::ostringstream os;
    os << static_cast<int>(d.architecture) << "|" << localimage::models::architectureName(d.architecture)
       << "|" << info.native_width << "x" << info.native_height;
    return env->NewStringUTF(os.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetModelHash(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    const auto* data = file->mappedData();
    if (!data) { throwJava(env, "java/lang/IllegalStateException", "model is not mapped"); return nullptr; }
    ModelHash hash;
    std::string error;
    if (!hash.compute(data, file->fileSize(), error)) { throwJava(env, "java/io/IOException", error); return nullptr; }
    return env->NewStringUTF(hash.hex().c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetVulkanMemoryBytes(JNIEnv* env, jclass) {
    localimage::VulkanContext context;
    std::string error;
    if (!context.initialize(error)) {
        throwJava(env, "java/io/IOException", "Vulkan memory query failed: " + error);
        return 0;
    }
    const uint64_t bytes = context.deviceInfo().device_local_heap_bytes;
    if (bytes > static_cast<uint64_t>(std::numeric_limits<jlong>::max())) return static_cast<jlong>(std::numeric_limits<jlong>::max());
    return static_cast<jlong>(bytes);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetDeviceInfo(JNIEnv* env, jclass) {
    localimage::VulkanContext context;
    std::string error;
    if (!context.initialize(error)) {
        throwJava(env, "java/io/IOException", "Vulkan device query failed: " + error);
        return nullptr;
    }
    const std::string summary = context.deviceInfo().summary();
    return env->NewStringUTF(summary.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetCacheKey(JNIEnv* env, jclass, jlong handle) {
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    const auto* data = file->mappedData();
    if (!data) { throwJava(env, "java/lang/IllegalStateException", "model is not mapped"); return nullptr; }

    ModelHash model_hash;
    std::string error;
    if (!model_hash.compute(data, file->fileSize(), error)) { throwJava(env, "java/io/IOException", error); return nullptr; }

    localimage::VulkanContext context;
    if (!context.initialize(error)) {
        throwJava(env, "java/io/IOException", "Vulkan device query failed: " + error);
        return nullptr;
    }

    CacheKeyInput input;
    input.model_sha256 = model_hash.hex();
    input.model_version = "raw-safetensors";
    input.runtime_version = "0.10.0-m10-neural-runtime";
    input.shader_version = "vulkan-compute-m10";
    input.device = context.deviceInfo();

    const std::string key = CacheKey::build(input);
    if (key.empty()) { throwJava(env, "java/io/IOException", "failed to build cache key"); return nullptr; }
    return env->NewStringUTF(key.c_str());
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeRunTensorTest(JNIEnv* env, jclass, jstring tempDir) {
    std::ostringstream report;
    auto fail = [&](const std::string& e) -> jstring { report << "✕ " << e; return env->NewStringUTF(report.str().c_str()); };
    TensorRuntime rt;
    std::string error;

    Tensor a = rt.createTensor(TensorShape({2, 3}), TensorDType::F32, error);
    if (!error.empty() || !a.valid()) return fail("Tensor allocation: " + error);
    float* ap = static_cast<float*>(a.mutableData());
    for (int i=0;i<6;++i) ap[i]=static_cast<float>(i+1);
    if (a.shape().elementCount()!=6 || a.stride().stride(0)!=3 || a.stride().stride(1)!=1) return fail("shape/stride");
    report << "Tensor allocation ✓\nShape/Stride ✓\nF32 ✓\n";

    Tensor b = rt.createTensor(TensorShape({3}), TensorDType::F32, error);
    if (!error.empty()) return fail("broadcast allocation: " + error);
    float* bp = static_cast<float*>(b.mutableData()); bp[0]=10; bp[1]=20; bp[2]=30;
    Tensor c;
    if (!add(a,b,c,error)) return fail("Add/Broadcast: " + error);
    const float* cp=static_cast<const float*>(c.data());
    const float expected[6]={11,22,33,14,25,36};
    for(int i=0;i<6;++i) if(std::fabs(cp[i]-expected[i])>1e-6f) return fail("Add numerical mismatch");
    report << "Add ✓\nBroadcasting ✓\n";

    Tensor d;
    if (!sub(a,b,d,error) || !mul(a,b,d,error) || !div(a,b,d,error)) return fail("Sub/Mul/Div: " + error);
    report << "Sub ✓\nMul ✓\nDiv ✓\n";

    Tensor reshaped;
    if (!rt.reshape(a, TensorShape({3,2}), reshaped, error)) return fail("reshape: " + error);
    if (reshaped.shape().dim(0)!=3 || reshaped.shape().dim(1)!=2 || reshaped.data()!=a.data()) return fail("reshape view semantics");
    report << "Reshape ✓\n";

    Tensor sliced;
    if (!rt.slice(a, 0, 1, 1, sliced, error)) return fail("slice: " + error);
    const float* sp=static_cast<const float*>(sliced.data());
    if (sliced.shape().dim(0)!=1 || std::fabs(sp[0]-4)>1e-6f) return fail("slice result");
    report << "Slice ✓\n";

    TensorShape huge({UINT64_MAX, 2});
    if (huge.valid()) return fail("overflow protection");
    report << "Overflow ✓\n";

    // Build a tiny real SafeTensors file and verify TensorView points at mmap-backed bytes.
    const std::string header = R"({"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
    const char* dir = tempDir ? env->GetStringUTFChars(tempDir, nullptr) : "/tmp";
    if (!dir) return fail("invalid temporary directory");
    std::string tempPath = std::string(dir) + "/localimage_tensor_XXXXXX";
    std::vector<char> path(tempPath.begin(), tempPath.end()); path.push_back(0);
    if (tempDir) env->ReleaseStringUTFChars(tempDir, dir);
    int fd = mkstemp(path.data());
    if (fd < 0) return fail("mmap test temp file: mkstemp failed");
    const uint64_t hs = header.size();
    uint8_t len[8]; for(int i=0;i<8;++i) len[i]=static_cast<uint8_t>((hs>>(8*i))&0xff);
    const float vals[4]={1,2,3,4};
    bool io = write(fd,len,8)==8 && write(fd,header.data(),header.size())==static_cast<ssize_t>(header.size()) && write(fd,vals,sizeof(vals))==sizeof(vals);
    if (!io) { close(fd); unlink(path.data()); return fail("mmap test write failed"); }
    lseek(fd,0,SEEK_SET);
    SafeTensorFile sf; if(!sf.open(fd,error) || !sf.validate(error)) { unlink(path.data()); return fail("mmap SafeTensors: "+error); }
    TensorView tv; if(!sf.getTensorView("x",tv,error)) { unlink(path.data()); return fail("mmap TensorView: "+error); }
    Tensor mapped = rt.createView(tv,error);
    if(!error.empty() || mapped.device()!=TensorDevice::MAPPED || mapped.data()!=tv.data()) { unlink(path.data()); return fail("mmap zero-copy view"); }
    const float* mp=static_cast<const float*>(mapped.data());
    if(mp[0]!=1 || mp[3]!=4) { unlink(path.data()); return fail("mmap bytes mismatch"); }
    report << "mmap TensorView ✓\n";
    sf.close();
    if(mapped.data()==nullptr || static_cast<const float*>(mapped.data())[2]!=3) return fail("mapped Tensor lifetime");
    report << "Mapped lifetime ✓\n\nTensor Runtime tests passed";
    unlink(path.data());
    return env->NewStringUTF(report.str().c_str());
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeRunVulkanComputeTest(JNIEnv* env, jclass) {
    localimage::VulkanContext context; std::string error;
    if (!context.initialize(error)) return env->NewStringUTF((std::string("Vulkan Compute Failed\n") + error).c_str());
    localimage::vulkan::VulkanCompute compute(context);
    return env->NewStringUTF(compute.runTests().report.c_str());
}



extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeResolveResolution(
        JNIEnv* env, jclass, jint architecture, jint width, jint height,
        jint dtype, jint backend, jlong weightBytes, jlong cpuBytes, jlong gpuBytes, jlong npuBytes) {
    using localimage::models::Architecture;
    using localimage::runtime::ResolutionPolicy;
    using localimage::runtime::ResolutionPlanner;
    using localimage::runtime::ResolutionRequest;
    using localimage::runtime::ResolvedResolution;
    using localimage::tensor::TensorDType;
    if (architecture < 0 || architecture > static_cast<jint>(Architecture::Anima)) {
        throwJava(env, "java/lang/IllegalArgumentException", "invalid architecture id");
        return nullptr;
    }
    if (width < 0 || height < 0 || weightBytes < 0 || cpuBytes < 0 || gpuBytes < 0 || npuBytes < 0) {
        throwJava(env, "java/lang/IllegalArgumentException", "resolution and memory values must be non-negative");
        return nullptr;
    }
    if (backend < 0 || backend > 3) { throwJava(env, "java/lang/IllegalArgumentException", "invalid resolution planner backend"); return nullptr; }
    TensorDType dt = TensorDType::F16;
    if (dtype == 0) dt = TensorDType::F32;
    else if (dtype == 1) dt = TensorDType::F16;
    else if (dtype == 2) dt = TensorDType::BF16;
    else { throwJava(env, "java/lang/IllegalArgumentException", "unsupported resolution planner dtype"); return nullptr; }
    const Architecture arch = static_cast<Architecture>(architecture);
    auto model = ResolutionPolicy::defaults(arch);
    model.weight_bytes = static_cast<uint64_t>(weightBytes);
    ResolutionRequest req;
    req.width = static_cast<uint32_t>(width); req.height = static_cast<uint32_t>(height); req.dtype = dt;
    req.backend = static_cast<localimage::runtime::ResolutionBackend>(backend);
    req.available_cpu_bytes = static_cast<uint64_t>(cpuBytes);
    req.available_gpu_bytes = static_cast<uint64_t>(gpuBytes);
    req.available_npu_bytes = static_cast<uint64_t>(npuBytes);
    ResolvedResolution r; std::string error;
    if (!ResolutionPlanner{}.resolve(model, req, r, error)) {
        throwJava(env, "java/lang/IllegalArgumentException", error);
        return nullptr;
    }
    std::ostringstream os;
    os << "requested=" << r.requested_width << "x" << r.requested_height
       << ";resolved=" << r.width << "x" << r.height
       << ";output=" << r.output_width << "x" << r.output_height
       << ";latent=" << r.latent_width << "x" << r.latent_height
       << ";alignment=" << r.alignment
       << ";estimatedPeakBytes=" << r.estimated_peak_bytes
       << ";memoryBudgetBytes=" << r.memory_budget_bytes
       << ";adjusted=" << (r.adjusted ? "true" : "false")
       << ";memoryLimited=" << (r.memory_limited ? "true" : "false")
       << ";reason=" << localimage::runtime::resolutionReasonName(r.reason);
    if (!r.warning.empty()) os << ";warning=" << r.warning;
    return env->NewStringUTF(os.str().c_str());
}

// ============================================================================
// NPU / Hexagon DSP detection
// ============================================================================

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGetNpuInfo(JNIEnv* env, jclass) {
    using localimage::npu::BackendProbe;
    using localimage::npu::Capabilities;

    Capabilities caps = BackendProbe::detect();

    std::ostringstream os;
    os << "SoC 型号: " << caps.socName << "\n";
    os << "DSP 版本: " << BackendProbe::dspVersionName(caps.dspVersion) << "\n";
    os << "后端类型: " << BackendProbe::backendName(caps.backend) << "\n";
    os << "设备: " << caps.device << "\n";
    os << "NPU 可用: " << (caps.available ? "是" : "否") << "\n";

    if (caps.totalMemoryBytes > 0) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        double mem = static_cast<double>(caps.totalMemoryBytes);
        int ui = 0;
        while (mem >= 1024 && ui < 3) { mem /= 1024; ui++; }
        os << "NPU 显存: " << std::fixed << std::setprecision(2) << mem << " " << units[ui] << "\n";
    }

    if (!caps.supportedOps.empty()) {
        os << "\n支持的算子 (" << caps.supportedOps.size() << " 种):\n";
        for (size_t i = 0; i < caps.supportedOps.size(); ++i) {
            if (i > 0) os << ", ";
            if (i > 0 && i % 6 == 0) os << "\n";
            os << caps.supportedOps[i];
        }
        os << "\n";
    }

    if (!caps.errorMessage.empty()) {
        os << "\n详细信息: " << caps.errorMessage << "\n";
    }

    if (!caps.available) {
        os << "\n最低要求: Snapdragon 8 Gen3 (Hexagon DSP v75)\n";
        os << "当前设备不满足 NPU 加速要求，将使用 Vulkan GPU / CPU 回退。";
    } else {
        os << "\n✓ NPU 加速已启用 (QNN HTP Backend)\n";
        os << "算子将优先在 Hexagon DSP 上执行，不支持的自动回退到 GPU/CPU。";
    }

    return env->NewStringUTF(os.str().c_str());
}
