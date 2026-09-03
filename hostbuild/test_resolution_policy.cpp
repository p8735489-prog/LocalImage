#include "../app/src/main/cpp/runtime/resolution_policy.h"
#include <cassert>
#include <iostream>

using namespace localimage;

static void expectDefault(models::Architecture arch, uint32_t w, uint32_t h) {
    runtime::ResolutionPlanner planner;
    runtime::ResolvedResolution r;
    std::string error;
    const auto model = runtime::ResolutionPolicy::defaults(arch);
    assert(planner.resolve(model, {0,0,tensor::TensorDType::F16,runtime::ResolutionBackend::Auto,0,0,0,false}, r, error));
    assert(r.output_width == w && r.output_height == h);
    assert(r.width % r.alignment == 0 && r.height % r.alignment == 0);
    assert(r.latent_width > 0 && r.latent_height > 0);
}

int main() {
    expectDefault(models::Architecture::StableDiffusion15, 512, 512);
    expectDefault(models::Architecture::StableDiffusion2, 768, 768);
    expectDefault(models::Architecture::SDXL, 1024, 1024);
    expectDefault(models::Architecture::SD3, 1080, 1080);
    expectDefault(models::Architecture::SD35, 1080, 1080);
    expectDefault(models::Architecture::FLUX, 1080, 1080);
    expectDefault(models::Architecture::Anima, 1080, 1080);

    runtime::ResolutionPlanner p;
    runtime::ResolvedResolution r;
    std::string e;
    auto x = runtime::ResolutionPolicy::defaults(models::Architecture::SDXL);

    assert(p.resolve(x, {1024,768,tensor::TensorDType::F16,runtime::ResolutionBackend::Auto,0,0,0,true}, r, e));
    assert(r.output_width == 1024 && r.output_height == 768);
    assert(r.width == 1024 && r.height == 768);
    assert(r.latent_width == 128 && r.latent_height == 96);

    // 1080 is a valid VAE-aligned final target, while SDXL's graph is
    // executed at the nearest latent/UNet-safe 64-aligned shape.
    assert(p.resolve(x, {1080,1080,tensor::TensorDType::F16,runtime::ResolutionBackend::Auto,0,0,0,true}, r, e));
    assert(r.output_width == 1080 && r.output_height == 1080);
    assert(r.width == 1088 && r.height == 1088);
    assert(r.latent_width == 136 && r.latent_height == 136);
    assert(r.adjusted);

    // 1081 is normalized to the nearest VAE-safe output target (1080), then
    // the graph uses the legal 1088 execution shape.
    assert(p.resolve(x, {1081,1081,tensor::TensorDType::F16,runtime::ResolutionBackend::Auto,0,0,0,true}, r, e));
    assert(r.output_width == 1080 && r.output_height == 1080);
    assert(r.width == 1088 && r.height == 1088);
    assert(r.adjusted);

    // Memory pressure is never silently ignored.
    const uint64_t tiny = 64ull * 1024ull;
    assert(!p.resolve(x, {1024,1024,tensor::TensorDType::F16,runtime::ResolutionBackend::CPU,
                          tiny,0,0,true}, r, e));
    assert(!e.empty());

    std::cout << "Resolution policy tests: PASS\n";
    return 0;
}
