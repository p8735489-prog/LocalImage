#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>
#include <limits.h>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include "stable-diffusion.h"
#include "safetensors/safe_tensor_file.h"

namespace fs = std::filesystem;
using localimage::safetensors::SafeTensorFile;

namespace {
std::mutex g_sd_mutex;

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* p = env->GetStringUTFChars(s, nullptr);
    std::string out = p ? p : "";
    if (p) env->ReleaseStringUTFChars(s, p);
    return out;
}

void throwJava(JNIEnv* env, const char* cls, const std::string& msg) {
    jclass c = env->FindClass(cls);
    if (c) env->ThrowNew(c, msg.c_str());
}

SafeTensorFile* fromHandle(jlong handle) {
    return reinterpret_cast<SafeTensorFile*>(static_cast<uintptr_t>(handle));
}

bool regularPathForFd(int fd, std::string& path) {
    char link[64];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    char target[PATH_MAX];
    const ssize_t n = ::readlink(link, target, sizeof(target) - 1);
    if (n <= 0) return false;
    target[n] = 0;
    struct stat st{};
    if (::stat(target, &st) != 0 || !S_ISREG(st.st_mode)) return false;
    path.assign(target, static_cast<size_t>(n));
    return true;
}

bool copyFdToPath(int fd, const std::string& path, std::string& error) {
    if (::lseek(fd, 0, SEEK_SET) < 0) { error = "model fd is not seekable: " + std::string(std::strerror(errno)); return false; }
    int out = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (out < 0) { error = "cannot create model staging file: " + std::string(std::strerror(errno)); return false; }
    char buf[1024 * 1024];
    while (true) {
        const ssize_t r = ::read(fd, buf, sizeof(buf));
        if (r == 0) break;
        if (r < 0) { error = "model staging read failed: " + std::string(std::strerror(errno)); ::close(out); return false; }
        ssize_t done = 0;
        while (done < r) {
            const ssize_t w = ::write(out, buf + done, static_cast<size_t>(r - done));
            if (w <= 0) { error = "model staging write failed: " + std::string(std::strerror(errno)); ::close(out); return false; }
            done += w;
        }
    }
    ::fsync(out);
    ::close(out);
    return true;
}

std::string chooseBackend() {
    const size_t bytes = sd_list_devices(nullptr, 0);
    std::string list(bytes + 1, '\0');
    if (bytes) sd_list_devices(list.data(), bytes + 1);
    std::istringstream in(list);
    std::string line;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        const std::string name = tab == std::string::npos ? line : line.substr(0, tab);
        const std::string desc = tab == std::string::npos ? line : line.substr(tab + 1);
        if (desc.find("Vulkan") != std::string::npos || name.find("vulkan") != std::string::npos) return name;
    }
    return "cpu";
}

bool writePng(const sd_image_t& img, const std::string& path, std::string& error) {
    if (!img.data || img.width == 0 || img.height == 0 || (img.channel != 1 && img.channel != 3 && img.channel != 4)) {
        error = "stable-diffusion.cpp returned an invalid image buffer";
        return false;
    }
    auto crc32 = [](const uint8_t* d, size_t n) {
        uint32_t c = 0xffffffffu;
        for (size_t i = 0; i < n; ++i) { c ^= d[i]; for (int k = 0; k < 8; ++k) c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0); }
        return c ^ 0xffffffffu;
    };
    auto be32 = [](std::vector<uint8_t>& o, uint32_t x) { o.push_back(x >> 24); o.push_back(x >> 16); o.push_back(x >> 8); o.push_back(x); };
    auto chunk = [&](std::vector<uint8_t>& o, const char* type, const std::vector<uint8_t>& data) {
        const size_t at = o.size(); be32(o, static_cast<uint32_t>(data.size()));
        o.insert(o.end(), type, type + 4); o.insert(o.end(), data.begin(), data.end());
        be32(o, crc32(o.data() + at + 4, data.size() + 4));
    };
    const uint32_t w = img.width, h = img.height; const uint8_t channels = img.channel;
    const size_t rowBytes = static_cast<size_t>(w) * channels;
    std::vector<uint8_t> raw((rowBytes + 1) * h);
    for (uint32_t y = 0; y < h; ++y) {
        raw[static_cast<size_t>(y) * (rowBytes + 1)] = 0;
        std::memcpy(raw.data() + static_cast<size_t>(y) * (rowBytes + 1) + 1,
                    img.data + static_cast<size_t>(y) * rowBytes, rowBytes);
    }
    std::vector<uint8_t> z{0x78, 0x01};
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        const bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0);
        const uint16_t nn = static_cast<uint16_t>(n), inv = static_cast<uint16_t>(~nn);
        z.push_back(nn); z.push_back(nn >> 8); z.push_back(inv); z.push_back(inv >> 8);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n); pos += n;
    }
    uint32_t a = 1, b = 0; for (uint8_t x : raw) { a = (a + x) % 65521; b = (b + a) % 65521; }
    be32(z, (b << 16) | a);
    std::vector<uint8_t> out{137,80,78,71,13,10,26,10}, ihdr;
    be32(ihdr, w); be32(ihdr, h); ihdr.push_back(8); ihdr.push_back(channels == 1 ? 0 : 2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    chunk(out, "IHDR", ihdr); chunk(out, "IDAT", z); chunk(out, "IEND", {});
    std::ofstream f(path, std::ios::binary); if (!f) { error = "cannot create output PNG"; return false; }
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f) { error = "failed writing output PNG"; return false; }
    return true;
}

bool generate(SafeTensorFile& file, const std::string& prompt, const std::string& negative,
              int width, int height, int steps, float cfg, int64_t seed,
              const std::string& scheduler, const std::string& sampler,
              const std::string& output, std::string& result) {
    if (prompt.empty()) { result = "生成失败：提示词为空"; return false; }
    if (width < 64 || height < 64 || width % 8 || height % 8) { result = "生成失败：宽高必须 ≥64 且为 8 的倍数"; return false; }
    if (steps < 1 || steps > 150) { result = "生成失败：Steps 必须在 1~150"; return false; }
    if (cfg < 0.0f || cfg > 30.0f) { result = "生成失败：CFG 超出范围"; return false; }

    const int fd = file.fileDescriptor();
    std::string modelPath;
    if (!regularPathForFd(fd, modelPath)) {
        const std::string staging = output + ".model.safetensors";
        if (!copyFdToPath(fd, staging, result)) return false;
        modelPath = staging;
    }

    sd_ctx_params_t cp{}; sd_ctx_params_init(&cp);
    cp.model_path = modelPath.c_str();
    cp.n_threads = std::max(1, sd_get_num_physical_cores());
    cp.wtype = SD_TYPE_F16;
    cp.rng_type = CPU_RNG;
    cp.sampler_rng_type = CPU_RNG;
    cp.enable_mmap = true;
    cp.flash_attn = true;
    cp.diffusion_flash_attn = true;
    const std::string backend = chooseBackend();
    cp.backend = backend.c_str();
    cp.params_backend = "cpu";
    cp.auto_fit = true;

    sd_ctx_t* ctx = new_sd_ctx(&cp);
    if (!ctx) { result = "生成失败：stable-diffusion.cpp 无法加载模型（可能是模型格式/内存/组件不完整）"; return false; }
    if (!sd_ctx_supports_image_generation(ctx)) { free_sd_ctx(ctx); result = "生成失败：该模型未被推理引擎识别为可生成图像模型"; return false; }

    sd_img_gen_params_t p{}; sd_img_gen_params_init(&p);
    p.prompt = prompt.c_str(); p.negative_prompt = negative.c_str(); p.width = width; p.height = height;
    p.sample_params.sample_steps = steps; p.sample_params.guidance.txt_cfg = cfg;
    p.sample_params.scheduler = str_to_scheduler(scheduler.c_str());
    p.sample_params.sample_method = str_to_sample_method(sampler.c_str());
    p.seed = seed;
    p.batch_count = 1;

    sd_image_t* images = nullptr; int count = 0;
    const bool ok = generate_image(ctx, &p, &images, &count);
    if (!ok || !images || count < 1) {
        free_sd_ctx(ctx); result = "生成失败：推理执行失败，请检查模型组件、分辨率、内存与后端日志"; return false;
    }
    std::string error;
    const bool saved = writePng(images[0], output, error);
    free_sd_images(images, count); free_sd_ctx(ctx);
    if (!saved) { result = "生成失败：" + error; return false; }
    if (modelPath == output + ".model.safetensors") ::unlink(modelPath.c_str());
    std::ostringstream os; os << "生成成功\n" << output << "\n" << width << " × " << height << "\nbackend=" << backend << "\nsteps=" << steps << "\nseed=" << seed;
    result = os.str(); return true;
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haobai_localimage_NativeRuntime_nativeGenerate(JNIEnv* env, jclass, jlong handle,
    jstring prompt, jstring negative, jint width, jint height, jint steps, jfloat cfg, jlong seed,
    jstring scheduler, jstring sampler, jstring outputPath) {
    std::lock_guard<std::mutex> lock(g_sd_mutex);
    auto* file = fromHandle(handle);
    if (!file) { throwJava(env, "java/lang/IllegalStateException", "invalid SafeTensor handle"); return nullptr; }
    std::string result;
    generate(*file, jstr(env, prompt), jstr(env, negative), width, height, steps, cfg, seed,
             jstr(env, scheduler), jstr(env, sampler), jstr(env, outputPath), result);
    return env->NewStringUTF(result.c_str());
}
