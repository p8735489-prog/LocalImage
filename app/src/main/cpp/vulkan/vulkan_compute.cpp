#include "vulkan_compute.h"
#include "add_shader_spirv.h"
#include "mul_shader_spirv.h"
#include "silu_shader_spirv.h"
#if __has_include("generated/add_spirv.h")
#include "generated/add_spirv.h"
#define LI_HAS_GENERATED_ADD 1
#endif
#if __has_include("generated/mul_spirv.h")
#include "generated/mul_spirv.h"
#define LI_HAS_GENERATED_MUL 1
#endif
#if __has_include("generated/silu_spirv.h")
#include "generated/silu_spirv.h"
#define LI_HAS_GENERATED_SILU 1
#endif
#if __has_include("generated/gelu_spirv.h")
#include "generated/gelu_spirv.h"
#define LI_HAS_GELU 1
#endif
#if __has_include("generated/sub_spirv.h")
#include "generated/sub_spirv.h"
#define LI_HAS_SUB 1
#endif
#if __has_include("generated/div_spirv.h")
#include "generated/div_spirv.h"
#define LI_HAS_DIV 1
#endif
#if __has_include("generated/exp_spirv.h")
#include "generated/exp_spirv.h"
#define LI_HAS_EXP 1
#endif
#if __has_include("generated/sqrt_spirv.h")
#include "generated/sqrt_spirv.h"
#define LI_HAS_SQRT 1
#endif
#if __has_include("generated/rsqrt_spirv.h")
#include "generated/rsqrt_spirv.h"
#define LI_HAS_RSQRT 1
#endif
#if __has_include("generated/clamp_spirv.h")
#include "generated/clamp_spirv.h"
#define LI_HAS_CLAMP 1
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#if __has_include("generated/matmul_spirv.h")
#include "generated/matmul_spirv.h"
#define LI_HAS_MATMUL 1
#endif
#if __has_include("generated/softmax_spirv.h")
#include "generated/softmax_spirv.h"
#define LI_HAS_SOFTMAX 1
#endif
#if __has_include("generated/layernorm_spirv.h")
#include "generated/layernorm_spirv.h"
#define LI_HAS_LAYERNORM 1
#endif
#if __has_include("generated/rmsnorm_spirv.h")
#include "generated/rmsnorm_spirv.h"
#define LI_HAS_RMSNORM 1
#endif
#if __has_include("generated/groupnorm_spirv.h")
#include "generated/groupnorm_spirv.h"
#define LI_HAS_GROUPNORM 1
#endif
#if __has_include("generated/conv2d_spirv.h")
#include "generated/conv2d_spirv.h"
#define LI_HAS_CONV2D 1
#endif
#if __has_include("generated/attention_spirv.h")
#include "generated/attention_spirv.h"
#define LI_HAS_ATTENTION 1
#endif
#if __has_include("generated/rope_spirv.h")
#include "generated/rope_spirv.h"
#define LI_HAS_ROPE 1
#endif

#if __has_include("generated/transpose_spirv.h")
#include "generated/transpose_spirv.h"
#define LI_HAS_TRANSPOSE 1
#endif
#if __has_include("generated/slice_spirv.h")
#include "generated/slice_spirv.h"
#define LI_HAS_SLICE 1
#endif
#if __has_include("generated/broadcast_spirv.h")
#include "generated/broadcast_spirv.h"
#define LI_HAS_BROADCAST 1
#endif
#if __has_include("generated/concat_spirv.h")
#include "generated/concat_spirv.h"
#define LI_HAS_CONCAT 1
#endif
#if __has_include("generated/upsample_spirv.h")
#include "generated/upsample_spirv.h"
#define LI_HAS_UPSAMPLE 1
#endif
namespace localimage::vulkan {
namespace {
std::string vr(VkResult r){return std::to_string((int)r);}
bool good(const tensor::Tensor&t){return t.valid()&&t.dtype()==tensor::TensorDType::F32&&t.isContiguous();}
struct MatmulPush{uint32_t M,K,N,batch,hasBias;};
struct SoftmaxPush{uint32_t rows,axisN,inner;};
struct NormPush{uint32_t rows,n;float eps;uint32_t hasBias;};
struct RmsPush{uint32_t rows,n;float eps;};
struct GroupPush{uint32_t N,C,H,W,groups;float eps;uint32_t hasBias;};
struct ConvPush{uint32_t N,Cin,H,Wd,Cout,KH,KW,OH,OW,stride,pad,dilation,groups,hasBias;};
struct AttnPush{uint32_t B,H,Qn,Kn,D,Vd;float scale;uint32_t hasMask;};
struct RopePush{uint32_t B,S,D;};
}

bool VulkanCompute::supported(std::string&e)const{if(!c_.device()){e="Vulkan device unavailable";return false;}const auto cap=c_.capabilities();if(cap.max_workgroup_x<64){e="Vulkan max compute workgroup X < 64";return false;}return true;}
bool VulkanCompute::supportsOperator(ir::Op op)const{
 if(op==ir::Op::Add||op==ir::Op::Mul||op==ir::Op::SiLU){
     // The legacy embedded shaders are only safe for exact multiples of their
     // 64-lane dispatch size; the runtime checks the tail before using them.
     return true;
 }
#if defined(LI_HAS_SUB)
 #if defined(LI_HAS_GELU)
 if(op==ir::Op::GELU)return true;
#endif
 if(op==ir::Op::Sub)return true;
#endif
#if defined(LI_HAS_DIV)
 if(op==ir::Op::Div)return true;
#endif
#if defined(LI_HAS_EXP)
 if(op==ir::Op::Exp)return true;
#endif
#if defined(LI_HAS_SQRT)
 if(op==ir::Op::Sqrt)return true;
#endif
#if defined(LI_HAS_RSQRT)
 if(op==ir::Op::Rsqrt)return true;
#endif
#if defined(LI_HAS_CLAMP)
 if(op==ir::Op::Clamp)return true;
#endif
#if defined(LI_HAS_MATMUL)
 if(op==ir::Op::MatMul||op==ir::Op::BatchedMatMul||op==ir::Op::Linear)return true;
#endif
#if defined(LI_HAS_SOFTMAX)
 if(op==ir::Op::Softmax)return true;
#endif
#if defined(LI_HAS_LAYERNORM)
 if(op==ir::Op::LayerNorm)return true;
#endif
#if defined(LI_HAS_RMSNORM)
 if(op==ir::Op::RMSNorm)return true;
#endif
#if defined(LI_HAS_GROUPNORM)
 if(op==ir::Op::GroupNorm)return true;
#endif
#if defined(LI_HAS_CONV2D)
 if(op==ir::Op::Conv2D)return true;
#endif
#if defined(LI_HAS_ATTENTION)
 if(op==ir::Op::Attention)return true;
#endif
#if defined(LI_HAS_ROPE)
 if(op==ir::Op::RoPE)return true;
#endif
#if defined(LI_HAS_TRANSPOSE)
 if(op==ir::Op::Transpose)return true;
#endif
#if defined(LI_HAS_SLICE)
 if(op==ir::Op::Slice)return true;
#endif
#if defined(LI_HAS_BROADCAST)
 if(op==ir::Op::Broadcast)return true;
#endif
#if defined(LI_HAS_CONCAT)
 if(op==ir::Op::Concat)return true;
#endif
#if defined(LI_HAS_UPSAMPLE)
 if(op==ir::Op::Upsample)return true;
#endif
 return false;
}
bool VulkanCompute::layout(VkDescriptorSetLayout&o,uint32_t count,std::string&e){if(count==0||count>8){e="invalid Vulkan descriptor count";return false;}std::vector<VkDescriptorSetLayoutBinding>b(count);for(uint32_t i=0;i<count;++i)b[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};ci.bindingCount=count;ci.pBindings=b.data();VkResult r=vkCreateDescriptorSetLayout(c_.device(),&ci,nullptr,&o);if(r!=VK_SUCCESS)e="vkCreateDescriptorSetLayout: "+vr(r);return r==VK_SUCCESS;}

bool VulkanCompute::runElementwise(const tensor::Tensor& a, const tensor::Tensor* b,
                                     tensor::Tensor& out, int kind, double& gpu,
                                     std::string& e) {
    if (!good(a) || (b && !good(*b))) {
        e = "Vulkan elementwise requires contiguous F32 tensors";
        return false;
    }
    if (b && a.shape().dims() != b->shape().dims()) {
        e = "Vulkan elementwise currently requires equal shapes";
        return false;
    }
    if (kind == 4 && b) {
        const float* d = static_cast<const float*>(b->data());
        for (uint64_t i = 0; i < b->shape().elementCount(); ++i) {
            if (d[i] == 0.0f) { e = "division by zero"; return false; }
        }
    }
    const uint64_t n64 = a.shape().elementCount();
    if (n64 == 0 || n64 > UINT32_MAX) {
        e = "Vulkan elementwise tensor size exceeds shader dispatch range";
        return false;
    }

    tensor::TensorRuntime rt;
    out = rt.createTensor(a.shape(), tensor::TensorDType::F32, e);
    if (!out.valid()) return false;

    VulkanTensor A, B, C;
    VkQueue q = c_.computeQueue();
    if (!A.upload(c_.physicalDevice(), c_.device(), c_.computeQueueFamily(), a, q, e)) return false;
    if (b && !B.upload(c_.physicalDevice(), c_.device(), c_.computeQueueFamily(), *b, q, e)) return false;
    if (!C.allocate(c_.physicalDevice(), c_.device(), c_.computeQueueFamily(),
                    a.shape(), tensor::TensorDType::F32, e)) return false;

    const uint32_t* code = nullptr;
    size_t words = 0;
    bool bounded = false;
    switch (kind) {
        case 0:
#if defined(LI_HAS_GENERATED_ADD)
            code = shader::li_add; words = shader::li_add_words; bounded = true;
#else
            code = shader::add; words = shader::add_words;
#endif
            break;
        case 1:
#if defined(LI_HAS_GENERATED_MUL)
            code = shader::li_mul; words = shader::li_mul_words; bounded = true;
#else
            code = shader::mul; words = shader::mul_words;
#endif
            break;
        case 2:
#if defined(LI_HAS_GENERATED_SILU)
            code = shader::li_silu; words = shader::li_silu_words; bounded = true;
#else
            code = shader::silu; words = shader::silu_words;
#endif
            break;
#if defined(LI_HAS_GELU)
        case 9: code = shader::li_gelu; words = shader::li_gelu_words; bounded = true; break;
#endif
#if defined(LI_HAS_SUB)
        case 3: code = shader::li_sub; words = shader::li_sub_words; bounded = true; break;
#endif
#if defined(LI_HAS_DIV)
        case 4: code = shader::li_div; words = shader::li_div_words; bounded = true; break;
#endif
#if defined(LI_HAS_EXP)
        case 5: code = shader::li_exp; words = shader::li_exp_words; bounded = true; break;
#endif
#if defined(LI_HAS_SQRT)
        case 6: code = shader::li_sqrt; words = shader::li_sqrt_words; bounded = true; break;
#endif
#if defined(LI_HAS_RSQRT)
        case 7: code = shader::li_rsqrt; words = shader::li_rsqrt_words; bounded = true; break;
#endif
#if defined(LI_HAS_CLAMP)
        case 8: code = shader::li_clamp; words = shader::li_clamp_words; bounded = true; break;
#endif
        default:
            e = "Vulkan elementwise operator unavailable";
            return false;
    }
    // Legacy embedded M5 shaders predate the bounds push constant. Never use
    // them for a partial final workgroup.
    if (!bounded && (n64 % 64u) != 0u) {
        e = "Vulkan operator unavailable: bounded elementwise shader not generated";
        return false;
    }

    VulkanShader sh;
    if (!sh.create(c_.device(), code, words, e)) return false;
    VkDescriptorSetLayout dl = VK_NULL_HANDLE;
    const uint32_t desc = b ? 3u : 2u;
    if (!layout(dl, desc, e)) return false;
    const uint32_t pushBytes = bounded ? (kind == 8 ? 12u : 4u) : 0u;
    VulkanPipeline pl;
    if (!pl.createCompute(c_.device(), sh.handle(), dl, pushBytes, e)) {
        vkDestroyDescriptorSetLayout(c_.device(), dl, nullptr);
        return false;
    }

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, desc};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkResult r = vkCreateDescriptorPool(c_.device(), &pci, nullptr, &pool);
    if (r != VK_SUCCESS) {
        e = "vkCreateDescriptorPool: " + vr(r);
        pl.destroy(); vkDestroyDescriptorSetLayout(c_.device(), dl, nullptr); return false;
    }
    VkDescriptorSet set{};
    VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sai.descriptorPool = pool; sai.descriptorSetCount = 1; sai.pSetLayouts = &dl;
    r = vkAllocateDescriptorSets(c_.device(), &sai, &set);
    if (r != VK_SUCCESS) {
        e = "vkAllocateDescriptorSets: " + vr(r);
        vkDestroyDescriptorPool(c_.device(), pool, nullptr); pl.destroy();
        vkDestroyDescriptorSetLayout(c_.device(), dl, nullptr); return false;
    }
    VkDescriptorBufferInfo bi[3] = {
        {A.buffer(), 0, A.byteSize()},
        {b ? B.buffer() : A.buffer(), 0, b ? B.byteSize() : A.byteSize()},
        {C.buffer(), 0, C.byteSize()}
    };
    std::vector<VkWriteDescriptorSet> writes(desc);
    for (uint32_t i = 0; i < desc; ++i)
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bi[i];
    vkUpdateDescriptorSets(c_.device(), desc, writes.data(), 0, nullptr);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = c_.computeQueueFamily();
    r = vkCreateCommandPool(c_.device(), &cpi, nullptr, &cp);
    VkCommandBuffer cb{};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    if (r == VK_SUCCESS) r = vkAllocateCommandBuffers(c_.device(), &cai, &cb);
    VkFence f = VK_NULL_HANDLE;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (r == VK_SUCCESS) r = vkCreateFence(c_.device(), &fi, nullptr, &f);
    if (r == VK_SUCCESS) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = vkBeginCommandBuffer(cb, &begin);
        if (r == VK_SUCCESS) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.handle());
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout(), 0, 1, &set, 0, nullptr);
            if (bounded) {
                if (kind == 8) {
                    struct { uint32_t n; float lo; float hi; } pc{static_cast<uint32_t>(n64), -1.0f, 1.0f};
                    vkCmdPushConstants(cb, pl.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                } else {
                    const uint32_t n = static_cast<uint32_t>(n64);
                    vkCmdPushConstants(cb, pl.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
                }
            }
            vkCmdDispatch(cb, static_cast<uint32_t>((n64 + 63u) / 64u), 1, 1);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cb);
        }
    }
    if (r == VK_SUCCESS) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        r = vkQueueSubmit(q, 1, &si, f);
        if (r == VK_SUCCESS) r = vkWaitForFences(c_.device(), 1, &f, VK_TRUE, UINT64_MAX);
    }
    if (r != VK_SUCCESS) e = "Vulkan elementwise execution: " + vr(r);
    if (r == VK_SUCCESS && !C.download(q, out, e)) r = VK_ERROR_UNKNOWN;
    if (f) vkDestroyFence(c_.device(), f, nullptr);
    if (cp) vkDestroyCommandPool(c_.device(), cp, nullptr);
    vkDestroyDescriptorPool(c_.device(), pool, nullptr);
    pl.destroy(); vkDestroyDescriptorSetLayout(c_.device(), dl, nullptr);
    return r == VK_SUCCESS && e.empty();
}

bool VulkanCompute::runGenerated(ir::Op op,const std::vector<tensor::Tensor>&in,const ir::Attributes&a,tensor::Tensor&out,double&gpu,std::string&e){for(const auto&t:in)if(!good(t)){e="Vulkan neural operators currently require contiguous F32 tensors";return false;}const uint32_t desc =
        (op==ir::Op::LayerNorm || op==ir::Op::GroupNorm || op==ir::Op::Conv2D) ? 4u :
        (op==ir::Op::RMSNorm ? 3u :
        (op==ir::Op::Attention ? 5u :
        (op==ir::Op::Softmax ? 2u :
        (op==ir::Op::RoPE ? 3u :
        (op==ir::Op::Linear ? 4u : 3u)))));if(in.size()>desc){e="Vulkan operator input count exceeds shader bindings";return false;}std::vector<VulkanTensor> gt(in.size());for(size_t i=0;i<in.size();++i)if(!gt[i].upload(c_.physicalDevice(),c_.device(),c_.computeQueueFamily(),in[i],c_.computeQueue(),e))return false;tensor::TensorShape shape;
 uint32_t pushBytes=0;std::vector<uint8_t> push;
#if defined(LI_HAS_MATMUL)
 if(op==ir::Op::MatMul||op==ir::Op::BatchedMatMul||op==ir::Op::Linear){size_t r=in[0].shape().rank(), br=in[1].shape().rank();if(r<2||br<2||r!=br){e="Vulkan MatMul requires equal rank >=2";return false;}size_t M=in[0].shape().dim(r-2),K=in[0].shape().dim(r-1),N=in[1].shape().dim(br-1);if(K!=in[1].shape().dim(br-2)){e="Vulkan MatMul K mismatch";return false;}uint32_t batch=1;for(size_t i=0;i+2<r;++i){if(in[0].shape().dim(i)!=in[1].shape().dim(i)){e="Vulkan MatMul batch broadcast requires materialization to equal batch shapes";return false;}if(in[0].shape().dim(i)>UINT32_MAX||batch>UINT32_MAX/in[0].shape().dim(i)){e="Vulkan MatMul batch overflow";return false;}batch*=static_cast<uint32_t>(in[0].shape().dim(i));}shape=TensorShape(r==2?std::vector<uint64_t>{M,N}:([&](){auto d=in[0].shape().dims();d[r-2]=M;d[r-1]=N;return d;})());MatmulPush pc{(uint32_t)M,(uint32_t)K,(uint32_t)N,batch,(uint32_t)(op==ir::Op::Linear&&in.size()==3)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));}
#endif
#if defined(LI_HAS_SOFTMAX)
 if(op==ir::Op::Softmax){if(in.size()!=1||in[0].shape().rank()==0){e="Vulkan Softmax requires one non-scalar input";return false;}size_t axis=a.axes.empty()?in[0].shape().rank()-1:a.axes[0];if(axis>=in[0].shape().rank()){e="Vulkan Softmax axis out of range";return false;}uint64_t inner=1;for(size_t i=axis+1;i<in[0].shape().rank();++i)inner*=in[0].shape().dim(i);uint64_t axisN=in[0].shape().dim(axis),rows=in[0].shape().elementCount()/(axisN*inner);if(axisN>UINT32_MAX||inner>UINT32_MAX||rows>UINT32_MAX){e="Vulkan Softmax dimensions exceed shader limits";return false;}SoftmaxPush pc{(uint32_t)rows,(uint32_t)axisN,(uint32_t)inner};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=in[0].shape();}
#endif
#if defined(LI_HAS_LAYERNORM)
 if(op==ir::Op::LayerNorm){if(in.size()<2||in.size()>3){e="Vulkan LayerNorm requires X,Gamma and optional Beta";return false;}size_t n=in[1].shape().dim(0);if(in[1].shape().rank()!=1||n!=in[0].shape().dim(in[0].shape().rank()-1)){e="Vulkan LayerNorm shape mismatch";return false;}uint64_t rows=in[0].shape().elementCount()/n;if(rows>UINT32_MAX||n>UINT32_MAX){e="Vulkan LayerNorm dimensions exceed shader limits";return false;}NormPush pc{(uint32_t)rows,(uint32_t)n,(float)a.epsilon,(uint32_t)(in.size()==3)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=in[0].shape();}
#endif
#if defined(LI_HAS_RMSNORM)
 if(op==ir::Op::RMSNorm){if(in.size()!=2||in[1].shape().rank()!=1||in[1].shape().dim(0)!=in[0].shape().dim(in[0].shape().rank()-1)){e="Vulkan RMSNorm shape mismatch";return false;}size_t n=in[1].shape().dim(0);uint64_t rows=in[0].shape().elementCount()/n;RmsPush pc{(uint32_t)rows,(uint32_t)n,(float)a.epsilon};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=in[0].shape();}
#endif
#if defined(LI_HAS_GROUPNORM)
 if(op==ir::Op::GroupNorm){if(in.size()<2||in.size()>3||in[0].shape().rank()!=4){e="Vulkan GroupNorm requires NCHW X,Gamma and optional Beta";return false;}size_t N=in[0].shape().dim(0),C=in[0].shape().dim(1),H=in[0].shape().dim(2),W=in[0].shape().dim(3),G=a.groups;if(G==0||C%G||in[1].shape().rank()!=1||in[1].shape().dim(0)!=C){e="Vulkan GroupNorm shape/groups mismatch";return false;}GroupPush pc{(uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)W,(uint32_t)G,(float)a.epsilon,(uint32_t)(in.size()==3)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=in[0].shape();}
#endif
#if defined(LI_HAS_CONV2D)
 if(op==ir::Op::Conv2D){if(in.size()<2||in.size()>3||in[0].shape().rank()!=4||in[1].shape().rank()!=4){e="Vulkan Conv2D requires NCHW X and OIHW weight";return false;}size_t N=in[0].shape().dim(0),Cin=in[0].shape().dim(1),H=in[0].shape().dim(2),W=in[0].shape().dim(3),Cout=in[1].shape().dim(0),Kc=in[1].shape().dim(1),KH=in[1].shape().dim(2),KW=in[1].shape().dim(3),G=a.groups;if(G==0||Cin%G||Cout%G||Kc!=Cin/G){e="Vulkan Conv2D groups/channel mismatch";return false;}uint64_t effH=1+(uint64_t)a.dilation*(KH-1),effW=1+(uint64_t)a.dilation*(KW-1),numH=(uint64_t)H+2*a.padding,numW=(uint64_t)W+2*a.padding;if(numH<effH||numW<effW){e="Vulkan Conv2D kernel exceeds padded input";return false;}size_t OH=(numH-effH)/a.stride+1,OW=(numW-effW)/a.stride+1;shape=TensorShape({N,Cout,OH,OW});ConvPush pc{(uint32_t)N,(uint32_t)Cin,(uint32_t)H,(uint32_t)W,(uint32_t)Cout,(uint32_t)KH,(uint32_t)KW,(uint32_t)OH,(uint32_t)OW,(uint32_t)a.stride,(uint32_t)a.padding,(uint32_t)a.dilation,(uint32_t)G,(uint32_t)(in.size()==3)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));}
#endif
#if defined(LI_HAS_ATTENTION)
 if(op==ir::Op::Attention){if(in.size()!=3&&in.size()!=4){e="Vulkan Attention requires Q,K,V and optional mask";return false;}if(in[0].shape().rank()!=4||in[1].shape().rank()!=4||in[2].shape().rank()!=4){e="Vulkan Attention shader requires [B,H,S,D]";return false;}size_t B=in[0].shape().dim(0),H=in[0].shape().dim(1),Q=in[0].shape().dim(2),D=in[0].shape().dim(3),K=in[1].shape().dim(2),Vd=in[2].shape().dim(3);if(in[1].shape().dim(0)!=B||in[2].shape().dim(0)!=B||in[1].shape().dim(1)!=H||in[2].shape().dim(1)!=H||in[1].shape().dim(3)!=D||in[2].shape().dim(2)!=K){e="Vulkan Attention Q/K/V shape mismatch";return false;}double scale=a.attention_scale>0?a.attention_scale:1.0/std::sqrt((double)D);AttnPush pc{(uint32_t)B,(uint32_t)H,(uint32_t)Q,(uint32_t)K,(uint32_t)D,(uint32_t)Vd,(float)scale,(uint32_t)(in.size()==4)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=TensorShape({B,H,Q,Vd});}
#endif
#if defined(LI_HAS_ROPE)
 if(op==ir::Op::RoPE){if(in.size()!=2||in[0].shape().rank()!=3||in[1].shape().rank()!=2||in[1].shape().dim(0)!=in[0].shape().dim(1)||in[1].shape().dim(1)>in[0].shape().dim(2)/2){e="Vulkan RoPE expects X=[B,S,D] and angles=[S,D/2]";return false;}RopePush pc{(uint32_t)in[0].shape().dim(0),(uint32_t)in[0].shape().dim(1),(uint32_t)in[0].shape().dim(2)};pushBytes=sizeof(pc);push.resize(pushBytes);std::memcpy(push.data(),&pc,sizeof(pc));shape=in[0].shape();}
#endif
 if(shape.rank()==0){e="Vulkan operator has no supported shape path";return false;}tensor::TensorRuntime rt;out=rt.createTensor(shape,tensor::TensorDType::F32,e);if(!out.valid())return false;VulkanTensor O;if(!O.allocate(c_.physicalDevice(),c_.device(),c_.computeQueueFamily(),shape,tensor::TensorDType::F32,e))return false;
 const uint32_t*code=nullptr;size_t words=0;
 switch(op){
#if defined(LI_HAS_MATMUL)
  case ir::Op::MatMul:case ir::Op::BatchedMatMul:case ir::Op::Linear:code=shader::matmul;words=shader::matmul_words;break;
#endif
#if defined(LI_HAS_SOFTMAX)
  case ir::Op::Softmax:code=shader::softmax;words=shader::softmax_words;break;
#endif
#if defined(LI_HAS_LAYERNORM)
  case ir::Op::LayerNorm:code=shader::layernorm;words=shader::layernorm_words;break;
#endif
#if defined(LI_HAS_RMSNORM)
  case ir::Op::RMSNorm:code=shader::rmsnorm;words=shader::rmsnorm_words;break;
#endif
#if defined(LI_HAS_GROUPNORM)
  case ir::Op::GroupNorm:code=shader::groupnorm;words=shader::groupnorm_words;break;
#endif
#if defined(LI_HAS_CONV2D)
  case ir::Op::Conv2D:code=shader::conv2d;words=shader::conv2d_words;break;
#endif
#if defined(LI_HAS_ATTENTION)
  case ir::Op::Attention:code=shader::attention;words=shader::attention_words;break;
#endif
#if defined(LI_HAS_ROPE)
  case ir::Op::RoPE:code=shader::rope;words=shader::rope_words;break;
#endif
  default:e="Vulkan operator unavailable: "+std::string(ir::opName(op));return false;
 }
 VulkanShader sh;if(!sh.create(c_.device(),code,words,e))return false;VkDescriptorSetLayout dl=VK_NULL_HANDLE;if(!layout(dl,desc,e))return false;VulkanPipeline pl;if(!pl.createCompute(c_.device(),sh.handle(),dl,pushBytes,e)){vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}
 VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,desc};VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pci.maxSets=1;pci.poolSizeCount=1;pci.pPoolSizes=&ps;VkDescriptorPool pool=VK_NULL_HANDLE;VkResult r=vkCreateDescriptorPool(c_.device(),&pci,nullptr,&pool);if(r!=VK_SUCCESS){e="vkCreateDescriptorPool: "+vr(r);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}VkDescriptorSet set{};VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};sai.descriptorPool=pool;sai.descriptorSetCount=1;sai.pSetLayouts=&dl;r=vkAllocateDescriptorSets(c_.device(),&sai,&set);if(r!=VK_SUCCESS){e="vkAllocateDescriptorSets: "+vr(r);vkDestroyDescriptorPool(c_.device(),pool,nullptr);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}
 std::vector<VkDescriptorBufferInfo> bi(desc,{VK_NULL_HANDLE,0,0});for(size_t i=0;i<gt.size();++i)bi[i]={gt[i].buffer(),0,gt[i].byteSize()};for(uint32_t i=(uint32_t)gt.size();i<desc;++i)bi[i]=bi[0];if(op==ir::Op::MatMul||op==ir::Op::BatchedMatMul){bi[2]={O.buffer(),0,O.byteSize()};}else if(op==ir::Op::Linear){bi[2]={O.buffer(),0,O.byteSize()};if(gt.size()<3)bi[3]=bi[0];}else{bi[desc-1]={O.buffer(),0,O.byteSize()};if(op==ir::Op::LayerNorm&&gt.size()==2)bi[2]=bi[0];if(op==ir::Op::GroupNorm&&gt.size()==2)bi[2]=bi[0];if(op==ir::Op::Conv2D&&gt.size()==2)bi[2]=bi[0];if(op==ir::Op::Attention&&gt.size()==3)bi[3]=bi[0];}
 std::vector<VkWriteDescriptorSet>w(desc);for(uint32_t i=0;i<desc;++i){w[i]={};w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=set;w[i].dstBinding=i;w[i].descriptorCount=1;w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[i].pBufferInfo=&bi[i];}vkUpdateDescriptorSets(c_.device(),desc,w.data(),0,nullptr);
 VkCommandPool cp=VK_NULL_HANDLE;VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=c_.computeQueueFamily();r=vkCreateCommandPool(c_.device(),&cpi,nullptr,&cp);VkCommandBuffer cb{};VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=cp;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;if(r==VK_SUCCESS)r=vkAllocateCommandBuffers(c_.device(),&cai,&cb);VkFence f=VK_NULL_HANDLE;VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};if(r==VK_SUCCESS)r=vkCreateFence(c_.device(),&fi,nullptr,&f);if(r==VK_SUCCESS){VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;r=vkBeginCommandBuffer(cb,&begin);if(r==VK_SUCCESS){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl.handle());vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl.layout(),0,1,&set,0,nullptr);if(pushBytes)vkCmdPushConstants(cb,pl.layout(),VK_SHADER_STAGE_COMPUTE_BIT,0,pushBytes,push.data());uint32_t gx=1,gy=1,gz=1;if(op==ir::Op::MatMul||op==ir::Op::BatchedMatMul||op==ir::Op::Linear){auto r0=in[0].shape().rank();gx=(uint32_t)((in[1].shape().dim(r0-1)+15)/16);gy=(uint32_t)((in[0].shape().dim(r0-2)+15)/16);uint32_t batch=1;for(size_t i=0;i+2<r0;++i)batch*=in[0].shape().dim(i);gz=batch;}else if(op==ir::Op::Softmax){uint32_t rows=0;std::memcpy(&rows,push.data(),sizeof(rows));gx=rows;}else if(op==ir::Op::LayerNorm||op==ir::Op::RMSNorm){uint32_t rows=0;std::memcpy(&rows,push.data(),sizeof(rows));gx=rows;}else if(op==ir::Op::GroupNorm){auto N=in[0].shape().dim(0);gx=(uint32_t)(N*a.groups);}else if(op==ir::Op::Conv2D){gx=(uint32_t)((shape.dim(3)+7)/8);gy=(uint32_t)((shape.dim(2)+7)/8);gz=(uint32_t)(shape.dim(0)*shape.dim(1));}else if(op==ir::Op::Attention){gx=(uint32_t)(in[0].shape().dim(0)*in[0].shape().dim(1)*in[0].shape().dim(2));}else if(op==ir::Op::RoPE){gx=(uint32_t)((in[0].shape().elementCount()/2+63)/64);}vkCmdDispatch(cb,gx,gy,gz);VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);r=vkEndCommandBuffer(cb);}}if(r==VK_SUCCESS){VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cb;r=vkQueueSubmit(c_.computeQueue(),1,&si,f);if(r==VK_SUCCESS)r=vkWaitForFences(c_.device(),1,&f,VK_TRUE,UINT64_MAX);}if(r!=VK_SUCCESS)e="Vulkan neural operator execution: "+vr(r);if(r==VK_SUCCESS&&!O.download(c_.computeQueue(),out,e)){}if(f)vkDestroyFence(c_.device(),f,nullptr);if(cp)vkDestroyCommandPool(c_.device(),cp,nullptr);vkDestroyDescriptorPool(c_.device(),pool,nullptr);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return e.empty();}


namespace {
bool dispatchResident(VulkanContext& c, const uint32_t* code, size_t words,
                      uint32_t descriptorCount, const void* push, uint32_t pushBytes,
                      uint32_t gx, uint32_t gy, uint32_t gz,
                      const std::vector<const VulkanTensor*>& inputs,
                      VulkanTensor& output, std::string& error) {
    if (!code || words == 0 || descriptorCount == 0 || descriptorCount > 8) {
        error = "invalid Vulkan resident dispatch configuration";
        return false;
    }
    if (!output.resident()) {
        error = "Vulkan resident dispatch requires an allocated output tensor";
        return false;
    }
    for (const VulkanTensor* t : inputs) {
        if (!t || !t->resident() || t->device() != c.device() || t->queueFamily() != c.computeQueueFamily()) {
            error = "Vulkan resident input belongs to another or invalid device";
            return false;
        }
    }
    if (output.device() != c.device() || output.queueFamily() != c.computeQueueFamily()) {
        error = "Vulkan resident output belongs to another device";
        return false;
    }
    VulkanShader shader;
    if (!shader.create(c.device(), code, words, error)) return false;
    VkDescriptorSetLayout dl = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings(descriptorCount);
    for (uint32_t i = 0; i < descriptorCount; ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = descriptorCount; dlci.pBindings = bindings.data();
    VkResult layoutResult = vkCreateDescriptorSetLayout(c.device(), &dlci, nullptr, &dl);
    if (layoutResult != VK_SUCCESS) { error = "vkCreateDescriptorSetLayout: " + vr(layoutResult); return false; }
    VulkanPipeline pipeline;
    if (!pipeline.createCompute(c.device(), shader.handle(), dl, pushBytes, error)) {
        vkDestroyDescriptorSetLayout(c.device(), dl, nullptr);
        return false;
    }
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkResult r = vkCreateDescriptorPool(c.device(), &pci, nullptr, &pool);
    if (r != VK_SUCCESS) {
        error = "vkCreateDescriptorPool: " + vr(r);
        pipeline.destroy(); vkDestroyDescriptorSetLayout(c.device(), dl, nullptr); return false;
    }
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sai.descriptorPool = pool; sai.descriptorSetCount = 1; sai.pSetLayouts = &dl;
    r = vkAllocateDescriptorSets(c.device(), &sai, &set);
    if (r != VK_SUCCESS) {
        error = "vkAllocateDescriptorSets: " + vr(r);
        vkDestroyDescriptorPool(c.device(), pool, nullptr); pipeline.destroy();
        vkDestroyDescriptorSetLayout(c.device(), dl, nullptr); return false;
    }
    std::vector<VkDescriptorBufferInfo> infos(descriptorCount);
    for (uint32_t i = 0; i < descriptorCount; ++i) {
        if (i < inputs.size()) infos[i] = {inputs[i]->buffer(), 0, inputs[i]->byteSize()};
        else infos[i] = {output.buffer(), 0, output.byteSize()};
    }
    // The last descriptor is always the output for all resident kernels used here.
    infos[descriptorCount - 1] = {output.buffer(), 0, output.byteSize()};
    std::vector<VkWriteDescriptorSet> writes(descriptorCount);
    for (uint32_t i = 0; i < descriptorCount; ++i)
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    vkUpdateDescriptorSets(c.device(), descriptorCount, writes.data(), 0, nullptr);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpi.queueFamilyIndex = c.computeQueueFamily();
    r = vkCreateCommandPool(c.device(), &cpi, nullptr, &cp);
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (r == VK_SUCCESS) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        r = vkAllocateCommandBuffers(c.device(), &cai, &cb);
    }
    VkFence fence = VK_NULL_HANDLE;
    if (r == VK_SUCCESS) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        r = vkCreateFence(c.device(), &fi, nullptr, &fence);
    }
    if (r == VK_SUCCESS) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = vkBeginCommandBuffer(cb, &begin);
        if (r == VK_SUCCESS) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &set, 0, nullptr);
            if (pushBytes) vkCmdPushConstants(cb, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
            vkCmdDispatch(cb, gx, gy, gz);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cb);
        }
    }
    if (r == VK_SUCCESS) {
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
        r = vkQueueSubmit(c.computeQueue(), 1, &submit, fence);
        if (r == VK_SUCCESS) r = vkWaitForFences(c.device(), 1, &fence, VK_TRUE, UINT64_MAX);
    }
    if (r != VK_SUCCESS) error = "Vulkan resident dispatch: " + vr(r);
    if (fence) vkDestroyFence(c.device(), fence, nullptr);
    if (cp) vkDestroyCommandPool(c.device(), cp, nullptr);
    vkDestroyDescriptorPool(c.device(), pool, nullptr);
    pipeline.destroy(); vkDestroyDescriptorSetLayout(c.device(), dl, nullptr);
    return r == VK_SUCCESS;
}
}

bool VulkanCompute::executeResident(ir::Op op,
                                     const std::vector<const VulkanTensor*>& inputs,
                                     const ir::Attributes& attr,
                                     VulkanTensor& output,
                                     std::string& error) {
    if (!supported(error)) return false;
    for (const VulkanTensor* t : inputs) {
        if (!t || !t->resident() || t->dtype() != tensor::TensorDType::F32 || !t->shape().valid()) {
            error = "Vulkan resident execution requires resident F32 tensors";
            return false;
        }
    }
    if (inputs.empty()) { error = "Vulkan resident execution requires inputs"; return false; }

    const uint64_t maxU32 = UINT32_MAX;
    if (op == ir::Op::Add || op == ir::Op::Sub || op == ir::Op::Mul || op == ir::Op::Div ||
        op == ir::Op::SiLU || op == ir::Op::Exp || op == ir::Op::Sqrt || op == ir::Op::Rsqrt || op == ir::Op::Clamp) {
        const bool unary = op == ir::Op::SiLU || op == ir::Op::Exp || op == ir::Op::Sqrt || op == ir::Op::Rsqrt || op == ir::Op::Clamp;
        if ((unary && inputs.size() != 1) || (!unary && inputs.size() != 2)) { error = "resident elementwise input count mismatch"; return false; }
        if (!unary && inputs[0]->shape().dims() != inputs[1]->shape().dims()) { error = "resident elementwise requires equal shapes"; return false; }
        const uint64_t n = inputs[0]->shape().elementCount();
        if (n == 0 || n > maxU32) { error = "resident elementwise dispatch range exceeded"; return false; }
        if (op == ir::Op::Div) {
            // The GPU shader must not silently generate Inf/NaN for a known zero denominator.
            tensor::Tensor host;
            if (!inputs[1]->download(c_.computeQueue(), host, error)) return false;
            const float* d = static_cast<const float*>(host.data());
            for (uint64_t i = 0; i < n; ++i) if (d[i] == 0.0f) { error = "division by zero"; return false; }
        }
        if (!output.resident() || output.shape().dims() != inputs[0]->shape().dims() || output.dtype() != tensor::TensorDType::F32) {
            if (!output.allocate(c_.physicalDevice(), c_.device(), c_.computeQueueFamily(), inputs[0]->shape(), tensor::TensorDType::F32, error)) return false;
        }
        const uint32_t* code = nullptr; size_t words = 0;
        switch (op) {
#if defined(LI_HAS_GENERATED_ADD)
            case ir::Op::Add: code = shader::li_add; words = shader::li_add_words; break;
#else
            case ir::Op::Add: code = shader::add; words = shader::add_words; break;
#endif
#if defined(LI_HAS_GENERATED_MUL)
            case ir::Op::Mul: code = shader::li_mul; words = shader::li_mul_words; break;
#else
            case ir::Op::Mul: code = shader::mul; words = shader::mul_words; break;
#endif
#if defined(LI_HAS_SUB)
            case ir::Op::Sub: code = shader::li_sub; words = shader::li_sub_words; break;
#endif
#if defined(LI_HAS_DIV)
            case ir::Op::Div: code = shader::li_div; words = shader::li_div_words; break;
#endif
#if defined(LI_HAS_EXP)
            case ir::Op::Exp: code = shader::li_exp; words = shader::li_exp_words; break;
#endif
#if defined(LI_HAS_SQRT)
            case ir::Op::Sqrt: code = shader::li_sqrt; words = shader::li_sqrt_words; break;
#endif
#if defined(LI_HAS_RSQRT)
            case ir::Op::Rsqrt: code = shader::li_rsqrt; words = shader::li_rsqrt_words; break;
#endif
#if defined(LI_HAS_CLAMP)
            case ir::Op::Clamp: code = shader::li_clamp; words = shader::li_clamp_words; break;
#endif
#if defined(LI_HAS_GENERATED_SILU)
            case ir::Op::SiLU: code = shader::li_silu; words = shader::li_silu_words; break;
#else
            case ir::Op::SiLU: code = shader::silu; words = shader::silu_words; break;
#endif
            default: error = "Vulkan operator unavailable: " + std::string(ir::opName(op)); return false;
        }
        uint32_t pcN = static_cast<uint32_t>(n);
        bool legacyUnbounded = false;
#if !defined(LI_HAS_GENERATED_ADD)
        if (op == ir::Op::Add) legacyUnbounded = true;
#endif
#if !defined(LI_HAS_GENERATED_MUL)
        if (op == ir::Op::Mul) legacyUnbounded = true;
#endif
#if !defined(LI_HAS_GENERATED_SILU)
        if (op == ir::Op::SiLU) legacyUnbounded = true;
#endif
        if (legacyUnbounded && (n % 64u) != 0u) { error = "Vulkan operator unavailable: bounded elementwise shader not generated"; return false; }
        struct ClampPc { uint32_t n; float lo; float hi; } clampPc{pcN, -1.0f, 1.0f};
        const void* push = op == ir::Op::Clamp ? static_cast<const void*>(&clampPc) : static_cast<const void*>(&pcN);
        const uint32_t pushSize = legacyUnbounded ? 0u : (op == ir::Op::Clamp ? sizeof(ClampPc) : sizeof(pcN));
        return dispatchResident(c_, code, words, unary ? 2u : 3u, push, pushSize,
                                (pcN + 63u) / 64u, 1, 1, inputs, output, error);
    }

    if (op == ir::Op::Transpose || op == ir::Op::Slice || op == ir::Op::Broadcast ||
        op == ir::Op::Concat || op == ir::Op::Upsample) {
        const auto &in0 = inputs[0]->shape();
        const size_t r = in0.rank();
        if (r == 0 || r > 4) { error = "Vulkan transform supports rank 1..4"; return false; }
        tensor::TensorShape outShape;
        std::vector<uint8_t> push;
        uint32_t gx = 1, gy = 1, gz = 1, desc = 2;
        const uint32_t *code = nullptr; size_t words = 0;
        struct TransformPush { uint32_t rank; uint32_t dims[4]; uint32_t aux[4]; uint32_t outDims[4]; };
        struct ConcatPush { uint32_t axis, rank, outDims[4], sizes[4]; };
        struct UpsamplePush { uint32_t N,C,H,W,scale; };
        if (op == ir::Op::Transpose) {
#if defined(LI_HAS_TRANSPOSE)
            if (attr.permutation.size() != r) { error="Vulkan Transpose requires a full permutation"; return false; }
            TransformPush pc{}; pc.rank=(uint32_t)r;
            std::vector<uint8_t> seen(r);
            auto od=std::vector<uint64_t>(r);
            for(size_t i=0;i<r;++i){
                if(attr.permutation[i]>=r || seen[attr.permutation[i]]) { error="Vulkan Transpose invalid permutation"; return false; }
                seen[attr.permutation[i]]=1; pc.perm[i]=(uint32_t)attr.permutation[i];
                pc.dims[i]=(uint32_t)in0.dim(i); od[i]=in0.dim(attr.permutation[i]); pc.outDims[i]=(uint32_t)od[i];
            }
            outShape=tensor::TensorShape(od); code=shader::transpose;words=shader::transpose_words;
            gx=(uint32_t)((outShape.elementCount()+63)/64);
            push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));
#else
            error="Vulkan operator unavailable: Transpose shader not generated";return false;
#endif
        } else if (op == ir::Op::Slice) {
#if defined(LI_HAS_SLICE)
            if(attr.slice_starts.size()!=r||attr.slice_lengths.size()!=r){error="Vulkan Slice requires starts/lengths";return false;}
            TransformPush pc{};pc.rank=(uint32_t)r;std::vector<uint64_t> od(r);
            for(size_t i=0;i<r;++i){if(attr.slice_starts[i]>in0.dim(i)||attr.slice_lengths[i]>in0.dim(i)-attr.slice_starts[i]){error="Vulkan Slice out of bounds";return false;}pc.dims[i]=(uint32_t)in0.dim(i);pc.aux[i]=(uint32_t)attr.slice_starts[i];pc.outDims[i]=(uint32_t)attr.slice_lengths[i];od[i]=attr.slice_lengths[i];}
            outShape=tensor::TensorShape(od);code=shader::slice;words=shader::slice_words;gx=(uint32_t)((outShape.elementCount()+63)/64);push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));
#else
            error="Vulkan operator unavailable: Slice shader not generated";return false;
#endif
        } else if (op == ir::Op::Broadcast) {
#if defined(LI_HAS_BROADCAST)
            if(attr.broadcast_shape.size()!=r){error="Vulkan Broadcast requires target shape";return false;}
            TransformPush pc{};pc.rank=(uint32_t)r;std::vector<uint64_t> od(r);
            for(size_t i=0;i<r;++i){if(attr.broadcast_shape[i]==0||(in0.dim(i)!=attr.broadcast_shape[i]&&in0.dim(i)!=1)){error="Vulkan Broadcast shape mismatch";return false;}pc.dims[i]=(uint32_t)in0.dim(i);pc.outDims[i]=(uint32_t)attr.broadcast_shape[i];od[i]=attr.broadcast_shape[i];}
            outShape=tensor::TensorShape(od);code=shader::broadcast;words=shader::broadcast_words;gx=(uint32_t)((outShape.elementCount()+63)/64);push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));
#else
            error="Vulkan operator unavailable: Broadcast shader not generated";return false;
#endif
        } else if (op == ir::Op::Upsample) {
#if defined(LI_HAS_UPSAMPLE)
            if(r!=4||attr.scale_factor==0||attr.scale_factor>UINT32_MAX){error="Vulkan Upsample requires NCHW and positive scale";return false;}
            auto d=in0.dims();for(size_t i=0;i<4;++i)if(d[i]>UINT32_MAX){error="Vulkan Upsample dimension exceeds shader range";return false;}
            d[2]*=attr.scale_factor;d[3]*=attr.scale_factor;outShape=tensor::TensorShape(d);
            UpsamplePush pc{(uint32_t)in0.dim(0),(uint32_t)in0.dim(1),(uint32_t)in0.dim(2),(uint32_t)in0.dim(3),(uint32_t)attr.scale_factor};
            code=shader::upsample;words=shader::upsample_words;gx=(uint32_t)((outShape.elementCount()+63)/64);push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));
#else
            error="Vulkan operator unavailable: Upsample shader not generated";return false;
#endif
        } else {
#if defined(LI_HAS_CONCAT)
            if(inputs.size()<2||inputs.size()>4||attr.axes.size()!=1||attr.axes[0]>=r){error="Vulkan Concat requires 2..4 inputs and a valid axis";return false;}
            const size_t axis=attr.axes[0];auto od=in0.dims();uint64_t sum=0;
            ConcatPush pc{};pc.axis=(uint32_t)axis;pc.rank=(uint32_t)r;
            for(size_t i=0;i<inputs.size();++i){if(inputs[i]->shape().rank()!=r){error="Vulkan Concat rank mismatch";return false;}for(size_t j=0;j<r;++j)if(j!=axis&&inputs[i]->shape().dim(j)!=in0.dim(j)){error="Vulkan Concat shape mismatch";return false;}if(inputs[i]->shape().dim(axis)>UINT32_MAX){error="Vulkan Concat dimension exceeds shader range";return false;}pc.sizes[i]=(uint32_t)inputs[i]->shape().dim(axis);if(sum>UINT64_MAX-inputs[i]->shape().dim(axis)){error="Vulkan Concat overflow";return false;}sum+=inputs[i]->shape().dim(axis);}
            od[axis]=sum;for(size_t i=0;i<r;++i){if(od[i]>UINT32_MAX){error="Vulkan Concat output dimension exceeds shader range";return false;}pc.outDims[i]=(uint32_t)od[i];}outShape=tensor::TensorShape(od);code=shader::concat;words=shader::concat_words;gx=(uint32_t)((outShape.elementCount()+63)/64);desc=5;push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));
#else
            error="Vulkan operator unavailable: Concat shader not generated";return false;
#endif
        }
        if(!outShape.valid()){error=outShape.error();return false;}
        if(!output.resident()||output.shape().dims()!=outShape.dims()||output.dtype()!=tensor::TensorDType::F32)
            if(!output.allocate(c_.physicalDevice(),c_.device(),c_.computeQueueFamily(),outShape,tensor::TensorDType::F32,error)) return false;
        std::vector<const VulkanTensor*> dispatchInputs=inputs;
        if(op==ir::Op::Concat) while(dispatchInputs.size()<4) dispatchInputs.push_back(inputs[0]);
        return dispatchResident(c_,code,words,desc,push.data(),(uint32_t)push.size(),gx,gy,gz,dispatchInputs,output,error);
    }

    if (op == ir::Op::MatMul || op == ir::Op::BatchedMatMul || op == ir::Op::Linear) {
#if !defined(LI_HAS_MATMUL)
        error = "Vulkan operator unavailable: MatMul shader not generated"; return false;
#else
        if (inputs.size() < 2 || inputs.size() > 3) { error = "resident MatMul input count mismatch"; return false; }
        const size_t ar = inputs[0]->shape().rank(), br = inputs[1]->shape().rank();
        if (ar < 2 || ar != br) { error = "resident MatMul requires equal rank >= 2"; return false; }
        const size_t M = inputs[0]->shape().dim(ar-2), K = inputs[0]->shape().dim(ar-1);
        if (K != inputs[1]->shape().dim(br-2)) { error = "resident MatMul K mismatch"; return false; }
        const size_t N = inputs[1]->shape().dim(br-1);
        uint64_t batch = 1;
        std::vector<uint64_t> dims = inputs[0]->shape().dims();
        for (size_t i = 0; i + 2 < ar; ++i) {
            if (inputs[0]->shape().dim(i) != inputs[1]->shape().dim(i)) { error = "Vulkan operator unavailable: MatMul batch broadcast is not supported by this kernel"; return false; }
            if (dims[i] == 0 || batch > maxU32 / dims[i]) { error = "resident MatMul batch overflow"; return false; }
            batch *= dims[i];
        }
        if (M > maxU32 || K > maxU32 || N > maxU32 || batch > maxU32) { error = "resident MatMul dimension exceeds shader range"; return false; }
        dims[ar-2] = M; dims[ar-1] = N;
        tensor::TensorShape shape(dims);
        if (!shape.valid()) { error = shape.error(); return false; }
        if (!output.resident() || output.shape().dims() != shape.dims() || output.dtype() != tensor::TensorDType::F32) {
            if (!output.allocate(c_.physicalDevice(), c_.device(), c_.computeQueueFamily(), shape, tensor::TensorDType::F32, error)) return false;
        }
        MatmulPush pc{static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), static_cast<uint32_t>(batch),
                      static_cast<uint32_t>(op == ir::Op::Linear && inputs.size() == 3)};
        // Linear uses descriptor 3 for A/B/output when no bias and 4 when bias is present.
        const uint32_t desc = op == ir::Op::Linear ? 4u : 3u;
        // dispatchResident treats the last descriptor as output, so for Linear the bias must be
        // placed before the output. Build a local descriptor dispatch explicitly below.
        if (op != ir::Op::Linear || inputs.size() != 3) {
            return dispatchResident(c_, shader::matmul, shader::matmul_words, 3, &pc, sizeof(pc),
                                    (static_cast<uint32_t>(N)+15u)/16u, (static_cast<uint32_t>(M)+15u)/16u,
                                    static_cast<uint32_t>(batch), inputs, output, error);
        }
        // Linear bias descriptor is binding 3 and output is binding 2; use a dedicated command setup.
        // This path intentionally reuses the same shader/pipeline and does not download the tensors.
        VulkanShader sh; if(!sh.create(c_.device(),shader::matmul,shader::matmul_words,error)) return false;
        VkDescriptorSetLayout dl=VK_NULL_HANDLE; if(!layout(dl,4,error)) return false;
        VulkanPipeline pl; if(!pl.createCompute(c_.device(),sh.handle(),dl,sizeof(pc),error)){vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pci.maxSets=1;pci.poolSizeCount=1;pci.pPoolSizes=&ps;VkDescriptorPool pool=VK_NULL_HANDLE;VkResult r=vkCreateDescriptorPool(c_.device(),&pci,nullptr,&pool);if(r!=VK_SUCCESS){error="vkCreateDescriptorPool: "+vr(r);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}
        VkDescriptorSet set=VK_NULL_HANDLE;VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};sai.descriptorPool=pool;sai.descriptorSetCount=1;sai.pSetLayouts=&dl;r=vkAllocateDescriptorSets(c_.device(),&sai,&set);if(r!=VK_SUCCESS){error="vkAllocateDescriptorSets: "+vr(r);vkDestroyDescriptorPool(c_.device(),pool,nullptr);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return false;}
        VkDescriptorBufferInfo bi[4]={{inputs[0]->buffer(),0,inputs[0]->byteSize()},{inputs[1]->buffer(),0,inputs[1]->byteSize()},{output.buffer(),0,output.byteSize()},{inputs[2]->buffer(),0,inputs[2]->byteSize()}};VkWriteDescriptorSet w[4]{};for(uint32_t i=0;i<4;++i){w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=set;w[i].dstBinding=i;w[i].descriptorCount=1;w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[i].pBufferInfo=&bi[i];}vkUpdateDescriptorSets(c_.device(),4,w,0,nullptr);
        VkCommandPool cp=VK_NULL_HANDLE;VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;cpi.queueFamilyIndex=c_.computeQueueFamily();r=vkCreateCommandPool(c_.device(),&cpi,nullptr,&cp);VkCommandBuffer cb=VK_NULL_HANDLE;VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=cp;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;if(r==VK_SUCCESS)r=vkAllocateCommandBuffers(c_.device(),&cai,&cb);VkFence f=VK_NULL_HANDLE;VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};if(r==VK_SUCCESS)r=vkCreateFence(c_.device(),&fi,nullptr,&f);if(r==VK_SUCCESS){VkCommandBufferBeginInfo bgn{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bgn.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;r=vkBeginCommandBuffer(cb,&bgn);if(r==VK_SUCCESS){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl.handle());vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl.layout(),0,1,&set,0,nullptr);vkCmdPushConstants(cb,pl.layout(),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);vkCmdDispatch(cb,(static_cast<uint32_t>(N)+15u)/16u,(static_cast<uint32_t>(M)+15u)/16u,static_cast<uint32_t>(batch));VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);r=vkEndCommandBuffer(cb);}}if(r==VK_SUCCESS){VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cb;r=vkQueueSubmit(c_.computeQueue(),1,&si,f);if(r==VK_SUCCESS)r=vkWaitForFences(c_.device(),1,&f,VK_TRUE,UINT64_MAX);}if(r!=VK_SUCCESS)error="Vulkan resident Linear dispatch: "+vr(r);if(f)vkDestroyFence(c_.device(),f,nullptr);if(cp)vkDestroyCommandPool(c_.device(),cp,nullptr);vkDestroyDescriptorPool(c_.device(),pool,nullptr);pl.destroy();vkDestroyDescriptorSetLayout(c_.device(),dl,nullptr);return r==VK_SUCCESS;
#endif
    }
    // The reduction/convolution/attention kernels below use the same resident
    // descriptor contract as their non-resident counterparts.  They deliberately
    // do not materialize GPU inputs on the CPU.
#if defined(LI_HAS_SOFTMAX) || defined(LI_HAS_LAYERNORM) || defined(LI_HAS_RMSNORM) || defined(LI_HAS_GROUPNORM) || defined(LI_HAS_CONV2D) || defined(LI_HAS_ATTENTION) || defined(LI_HAS_ROPE)
    const uint32_t* code = nullptr; size_t words = 0;
    uint32_t gx = 1, gy = 1, gz = 1;
    std::vector<uint8_t> push;
    uint32_t pushBytes = 0;
    uint32_t descriptorCount = 0;
    tensor::TensorShape shape;

    if (op == ir::Op::Softmax) {
#if defined(LI_HAS_SOFTMAX)
        if (inputs.size()!=1 || inputs[0]->shape().rank()==0) { error="resident Softmax input count/rank mismatch"; return false; }
        const size_t rank=inputs[0]->shape().rank();
        const size_t axis=attr.axes.empty()?rank-1:attr.axes[0];
        if(axis>=rank){error="resident Softmax axis out of range";return false;}
        uint64_t inner=1;
        for(size_t i=axis+1;i<rank;++i) inner*=inputs[0]->shape().dim(i);
        const uint64_t axisN=inputs[0]->shape().dim(axis);
        const uint64_t rows=inputs[0]->shape().elementCount()/(axisN*inner);
        if(rows==0||rows>UINT32_MAX||axisN>UINT32_MAX||inner>UINT32_MAX){error="resident Softmax dimensions exceed shader range";return false;}
        struct {uint32_t rows,axisN,inner;} pc{(uint32_t)rows,(uint32_t)axisN,(uint32_t)inner};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=inputs[0]->shape(); code=shader::softmax;words=shader::softmax_words;
        descriptorCount=2;gx=(uint32_t)rows;
#endif
    } else if (op == ir::Op::LayerNorm) {
#if defined(LI_HAS_LAYERNORM)
        if(inputs.size()<2||inputs.size()>3||inputs[1]->shape().rank()!=1){error="resident LayerNorm inputs";return false;}
        const uint64_t n=inputs[1]->shape().dim(0);
        if(inputs[0]->shape().rank()==0||n!=inputs[0]->shape().dim(inputs[0]->shape().rank()-1)){error="resident LayerNorm shape mismatch";return false;}
        const uint64_t rows=inputs[0]->shape().elementCount()/n;
        if(rows>UINT32_MAX||n>UINT32_MAX){error="resident LayerNorm dimensions exceed shader range";return false;}
        struct {uint32_t rows,n;float eps;uint32_t hasBias;} pc{(uint32_t)rows,(uint32_t)n,(float)attr.epsilon,(uint32_t)(inputs.size()==3)};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=inputs[0]->shape();code=shader::layernorm;words=shader::layernorm_words;descriptorCount=4;gx=(uint32_t)rows;
#endif
    } else if (op == ir::Op::RMSNorm) {
#if defined(LI_HAS_RMSNORM)
        if(inputs.size()!=2||inputs[1]->shape().rank()!=1){error="resident RMSNorm inputs";return false;}
        const uint64_t n=inputs[1]->shape().dim(0);
        if(inputs[0]->shape().rank()==0||n!=inputs[0]->shape().dim(inputs[0]->shape().rank()-1)){error="resident RMSNorm shape mismatch";return false;}
        const uint64_t rows=inputs[0]->shape().elementCount()/n;
        if(rows>UINT32_MAX||n>UINT32_MAX){error="resident RMSNorm dimensions exceed shader range";return false;}
        struct {uint32_t rows,n;float eps;} pc{(uint32_t)rows,(uint32_t)n,(float)attr.epsilon};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=inputs[0]->shape();code=shader::rmsnorm;words=shader::rmsnorm_words;descriptorCount=3;gx=(uint32_t)rows;
#endif
    } else if (op == ir::Op::GroupNorm) {
#if defined(LI_HAS_GROUPNORM)
        if(inputs.size()<2||inputs.size()>3||inputs[0]->shape().rank()!=4){error="resident GroupNorm inputs";return false;}
        const size_t N=inputs[0]->shape().dim(0),C=inputs[0]->shape().dim(1),H=inputs[0]->shape().dim(2),W=inputs[0]->shape().dim(3),G=attr.groups;
        if(G==0||C%G||inputs[1]->shape().rank()!=1||inputs[1]->shape().dim(0)!=C){error="resident GroupNorm shape/groups mismatch";return false;}
        if(N>UINT32_MAX||C>UINT32_MAX||H>UINT32_MAX||W>UINT32_MAX||G>UINT32_MAX){error="resident GroupNorm dimensions exceed shader range";return false;}
        struct {uint32_t N,C,H,W,groups;float eps;uint32_t hasBias;} pc{(uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)W,(uint32_t)G,(float)attr.epsilon,(uint32_t)(inputs.size()==3)};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=inputs[0]->shape();code=shader::groupnorm;words=shader::groupnorm_words;descriptorCount=4;gx=(uint32_t)(N*G);
#endif
    } else if (op == ir::Op::Conv2D) {
#if defined(LI_HAS_CONV2D)
        if(inputs.size()<2||inputs.size()>3||inputs[0]->shape().rank()!=4||inputs[1]->shape().rank()!=4){error="resident Conv2D inputs";return false;}
        const size_t N=inputs[0]->shape().dim(0),Cin=inputs[0]->shape().dim(1),H=inputs[0]->shape().dim(2),W=inputs[0]->shape().dim(3);
        const size_t Cout=inputs[1]->shape().dim(0),Kc=inputs[1]->shape().dim(1),KH=inputs[1]->shape().dim(2),KW=inputs[1]->shape().dim(3),G=attr.groups;
        if(G==0||attr.stride==0||attr.dilation==0||Cin%G||Cout%G||Kc!=Cin/G||KH==0||KW==0){error="resident Conv2D parameters/channel mismatch";return false;}
        uint64_t effH=1+(uint64_t)attr.dilation*(KH-1),effW=1+(uint64_t)attr.dilation*(KW-1);
        uint64_t numH=(uint64_t)H+2ULL*attr.padding,numW=(uint64_t)W+2ULL*attr.padding;
        if(numH<effH||numW<effW){error="resident Conv2D kernel exceeds padded input";return false;}
        size_t OH=(size_t)((numH-effH)/attr.stride+1),OW=(size_t)((numW-effW)/attr.stride+1);
        if(N>UINT32_MAX||Cin>UINT32_MAX||H>UINT32_MAX||W>UINT32_MAX||Cout>UINT32_MAX||KH>UINT32_MAX||KW>UINT32_MAX||OH>UINT32_MAX||OW>UINT32_MAX||attr.stride>UINT32_MAX||attr.padding>UINT32_MAX||attr.dilation>UINT32_MAX||G>UINT32_MAX){error="resident Conv2D dimensions exceed shader range";return false;}
        struct {uint32_t N,Cin,H,Wd,Cout,KH,KW,OH,OW,stride,pad,dilation,groups,hasBias;} pc{(uint32_t)N,(uint32_t)Cin,(uint32_t)H,(uint32_t)W,(uint32_t)Cout,(uint32_t)KH,(uint32_t)KW,(uint32_t)OH,(uint32_t)OW,(uint32_t)attr.stride,(uint32_t)attr.padding,(uint32_t)attr.dilation,(uint32_t)G,(uint32_t)(inputs.size()==3)};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=tensor::TensorShape({N,Cout,OH,OW});code=shader::conv2d;words=shader::conv2d_words;descriptorCount=4;
        gx=(uint32_t)((OW+7)/8);gy=(uint32_t)((OH+7)/8);gz=(uint32_t)(N*Cout);
#endif
    } else if (op == ir::Op::Attention) {
#if defined(LI_HAS_ATTENTION)
        if(inputs.size()!=3&&inputs.size()!=4){error="resident Attention inputs";return false;}
        if(inputs[0]->shape().rank()!=4||inputs[1]->shape().rank()!=4||inputs[2]->shape().rank()!=4){error="resident Attention requires [B,H,S,D]";return false;}
        const size_t B=inputs[0]->shape().dim(0),HH=inputs[0]->shape().dim(1),Q=inputs[0]->shape().dim(2),D=inputs[0]->shape().dim(3),K=inputs[1]->shape().dim(2),Vd=inputs[2]->shape().dim(3);
        if(inputs[1]->shape().dim(0)!=B||inputs[2]->shape().dim(0)!=B||inputs[1]->shape().dim(1)!=HH||inputs[2]->shape().dim(1)!=HH||inputs[1]->shape().dim(3)!=D||inputs[2]->shape().dim(2)!=K){error="resident Attention Q/K/V mismatch";return false;}
        if(B>UINT32_MAX||HH>UINT32_MAX||Q>UINT32_MAX||K>UINT32_MAX||D>UINT32_MAX||Vd>UINT32_MAX){error="resident Attention dimensions exceed shader range";return false;} if(inputs.size()==4){const auto& m=inputs[3]->shape();if(m.rank()!=4||m.dim(0)!=B||m.dim(1)!=HH||m.dim(2)!=Q||m.dim(3)!=K){error="resident Attention mask must be [B,H,Q,K]";return false;}}
        const float scale=(float)(attr.attention_scale>0?attr.attention_scale:1.0/std::sqrt((double)D));
        struct {uint32_t B,H,Qn,Kn,D,Vd;float scale;uint32_t hasMask;} pc{(uint32_t)B,(uint32_t)HH,(uint32_t)Q,(uint32_t)K,(uint32_t)D,(uint32_t)Vd,scale,(uint32_t)(inputs.size()==4)};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=tensor::TensorShape({B,HH,Q,Vd});code=shader::attention;words=shader::attention_words;descriptorCount=5;
        gx=(uint32_t)(B*HH*Q);
#endif
    } else if (op == ir::Op::RoPE) {
#if defined(LI_HAS_ROPE)
        if(inputs.size()!=2||inputs[0]->shape().rank()!=3||inputs[1]->shape().rank()!=2){error="resident RoPE inputs";return false;}
        const size_t B=inputs[0]->shape().dim(0),S=inputs[0]->shape().dim(1),D=inputs[0]->shape().dim(2);
        if(inputs[1]->shape().dim(0)!=S||inputs[1]->shape().dim(1)>D/2){error="resident RoPE angle shape mismatch";return false;}
        if(B>UINT32_MAX||S>UINT32_MAX||D>UINT32_MAX){error="resident RoPE dimensions exceed shader range";return false;}
        struct {uint32_t B,S,D;} pc{(uint32_t)B,(uint32_t)S,(uint32_t)D};
        push.resize(sizeof(pc));std::memcpy(push.data(),&pc,sizeof(pc));pushBytes=sizeof(pc);
        shape=inputs[0]->shape();code=shader::rope;words=shader::rope_words;descriptorCount=3;
        gx=(uint32_t)((inputs[0]->shape().elementCount()/2+63)/64);
#endif
    }
    if (code && descriptorCount && shape.valid()) {
        if(!output.resident() || output.shape().dims()!=shape.dims() || output.dtype()!=tensor::TensorDType::F32) {
            if(!output.allocate(c_.physicalDevice(),c_.device(),c_.computeQueueFamily(),shape,tensor::TensorDType::F32,error)) return false;
        }
        // Softmax and the reduction kernels have output as their last descriptor;
        // their optional inputs occupy the preceding bindings exactly as declared.
        return dispatchResident(c_,code,words,descriptorCount,push.empty()?nullptr:push.data(),pushBytes,gx,gy,gz,inputs,output,error);
    }
#endif
    error = "Vulkan resident operator unavailable: " + std::string(ir::opName(op));
    return false;
}

bool VulkanCompute::execute(ir::Op op,const std::vector<tensor::Tensor>&inputs,const ir::Attributes&attr,tensor::Tensor&output,std::string&error){
    if(!supported(error)) return false;
    int kind=-1;
    switch(op){
        case ir::Op::Add: kind=0; break; case ir::Op::Mul: kind=1; break; case ir::Op::SiLU: kind=2; break;
        case ir::Op::Sub: kind=3; break; case ir::Op::Div: kind=4; break; case ir::Op::Exp: kind=5; break;
        case ir::Op::Sqrt: kind=6; break; case ir::Op::Rsqrt: kind=7; break; case ir::Op::Clamp: kind=8; break; case ir::Op::GELU: kind=9; break;
        default: break;
    }
    if(kind>=0){
        const bool unary = kind>=5;
        if((unary && inputs.size()!=1) || (!unary && kind==2 && inputs.size()!=1) || (!unary && kind!=2 && inputs.size()!=2)){
            error="Vulkan elementwise input count mismatch"; return false;
        }
        double ms=-1;
        return runElementwise(inputs[0], inputs.size()==2?&inputs[1]:nullptr, output, kind, ms, error);
    }
    if(!supportsOperator(op)){error="Vulkan operator unavailable: "+std::string(ir::opName(op));return false;}
    double ms=-1; return runGenerated(op,inputs,attr,output,ms,error);
}

GpuTestResult VulkanCompute::runTests(){GpuTestResult z;std::string e;if(!supported(e)){z.report="Vulkan unavailable\n"+e;return z;}tensor::TensorRuntime rt;tensor::Tensor a=rt.createTensor(tensor::TensorShape({64}),tensor::TensorDType::F32,e),b=rt.createTensor(tensor::TensorShape({64}),tensor::TensorDType::F32,e);if(!a.valid()||!b.valid()){z.report="allocation failed: "+e;return z;}for(size_t i=0;i<64;++i){static_cast<float*>(a.mutableData())[i]=float(i)*.01f;static_cast<float*>(b.mutableData())[i]=1.f-float(i)*.001f;}tensor::Tensor o;double ms=-1;if(!execute(ir::Op::Add,{a,b},{},o,e)){z.report="Add failed: "+e;return z;}double mx=0;for(size_t i=0;i<64;++i)mx=std::max(mx,std::fabs(static_cast<const float*>(o.data())[i]-(static_cast<const float*>(a.data())[i]+static_cast<const float*>(b.data())[i])));z.max_error=mx;if(mx>1e-5){z.report="Add numerical mismatch";return z;}if(!execute(ir::Op::Mul,{a,b},{},o,e)){z.report="Mul failed: "+e;return z;}if(!execute(ir::Op::SiLU,{a},{},o,e)){z.report="SiLU failed: "+e;return z;}z.success=true;z.report="Vulkan smoke verified\nGPU: "+c_.deviceInfo().device_name+"\nAdd ✓\nMul ✓\nSiLU ✓\nCore numerical max error: "+std::to_string(z.max_error)+"\nGenerated neural shaders: "+std::string(supportsOperator(ir::Op::MatMul)?"available":"not generated");return z;}
}
