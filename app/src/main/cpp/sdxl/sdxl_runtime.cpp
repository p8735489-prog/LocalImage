#include "sdxl_runtime.h"
#include "../operators/cpu_operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <limits>

namespace localimage::sdxl {
using tensor::Tensor; using tensor::TensorDType; using tensor::TensorRuntime; using tensor::TensorShape;
namespace {
bool f32(const Tensor&t,std::string&e){if(!t.valid()||t.dtype()!=TensorDType::F32||!t.isContiguous()){e="SDXL component requires contiguous F32 tensor";return false;}return true;}
bool hasAny(const safetensors::SafeTensorFile& f,const std::vector<std::string>& keys){for(const auto& kv:f.tensors())for(const auto& k:keys)if(kv.first.find(k)!=std::string::npos)return true;return false;}
bool alloc(const TensorShape&s,Tensor&o,std::string&e){TensorRuntime rt;o=rt.createTensor(s,TensorDType::F32,e);return o.valid();}
}

bool SDXLTextEncoder::encode(const std::vector<uint32_t>& ids,const safetensors::SafeTensorFile& weights,const std::string& prefix,Tensor& sequence,Tensor* pooled,std::string& error) const {
    if(ids.empty()||ids.size()>77){error="SDXL tokenizer sequence must contain 1..77 tokens";return false;}
    // This executor deliberately requires the canonical CLIP embedding weights. It never fabricates embeddings.
    const bool embedding=hasAny(weights,{prefix+"text_model.embeddings.token_embedding.weight",prefix+"embeddings.token_embedding.weight"});
    const bool pos=hasAny(weights,{prefix+"text_model.embeddings.position_embedding.weight",prefix+"embeddings.position_embedding.weight"});
    const bool finalNorm=hasAny(weights,{prefix+"text_model.final_layer_norm.weight",prefix+"final_layer_norm.weight"});
    if(!embedding||!pos||!finalNorm){error="SDXL text encoder weights are incomplete for prefix '"+prefix+"'";return false;}
    error="SDXL CLIP transformer execution requires a weight-bound transformer block graph; raw checkpoint is not executable by this foundation yet";
    return false;
}

bool SDXLUNet::validateWeights(const safetensors::SafeTensorFile& weights,std::string& error) const {
    const bool input=hasAny(weights,{"input_blocks.0.0","down_blocks.0.resnets.0","down_blocks.0"});
    const bool middle=hasAny(weights,{"middle_block","mid_block"});
    const bool output=hasAny(weights,{"out.0","out.2","conv_out"});
    if(!input||!middle||!output){error="SDXL UNet checkpoint is missing canonical input/middle/output weights";return false;}
    return true;
}

bool SDXLUNet::predict(const Tensor& latent,const SDXLConditioning&,double,const safetensors::SafeTensorFile& weights,Tensor&,std::string& error) const {
    if(!f32(latent,error))return false;
    if(!validateWeights(weights,error))return false;
    error="SDXL UNet weight-bound residual/attention graph is not available in the current source tree; refusing to produce a fabricated noise prediction";
    return false;
}

bool SDXLVAE::validateWeights(const safetensors::SafeTensorFile& weights,std::string& error) const {
    if(!hasAny(weights,{"first_stage_model.decoder","vae.decoder","decoder.conv_in","decoder.mid_block"})){error="SDXL VAE decoder weights not found";return false;}
    return true;
}

bool SDXLVAE::decode(const Tensor& latent,const safetensors::SafeTensorFile& weights,Tensor& rgb,std::string& error) const {
    if(!f32(latent,error)||latent.shape().rank()!=4){error="SDXL VAE expects NCHW F32 latent";return false;}
    if(!validateWeights(weights,error))return false;
    error="SDXL VAE decoder weight-bound graph is not available in the current source tree; refusing to fabricate RGB output";
    return false;
}

bool SDXLRuntime::load(const safetensors::SafeTensorFile& weights,std::string& error){
    if(weights.tensors().empty()){error="cannot load empty SDXL checkpoint";return false;}
    // Fail closed unless all required component families are present.
    bool clip=false,openclip=false,unet=false,vae=false;
    for(const auto& kv:weights.tensors()){
        const auto& n=kv.first;
        clip|=n.find("conditioner.embedders.0")!=std::string::npos||n.find("text_encoder.")!=std::string::npos;
        openclip|=n.find("conditioner.embedders.1")!=std::string::npos||n.find("text_encoder_2.")!=std::string::npos;
        unet|=n.find("diffusion_model.")!=std::string::npos||n.find("model.diffusion_model.")!=std::string::npos||n.find("unet.")!=std::string::npos;
        vae|=n.find("first_stage_model.")!=std::string::npos||n.find("vae.")!=std::string::npos;
    }
    if(!clip||!openclip||!unet||!vae){error="SDXL load failed: checkpoint does not contain all required CLIP-L/OpenCLIP-G/UNet/VAE component families";return false;}
    if(!unet_.validateWeights(weights,error))return false;
    if(!vae_.validateWeights(weights,error))return false;
    weights_=&weights;
    return true;
}

bool SDXLRuntime::generate(const SDXLRequest& r,Tensor& rgb,std::string& error){
    if(!weights_){error="SDXL runtime is not loaded";return false;}
    if(r.steps==0||r.steps>200||r.cfg_scale<0||r.width<64||r.height<64||r.width%8||r.height%8){error="invalid SDXL generation parameters";return false;}
    error="SDXL generation cannot proceed until tokenizer assets and weight-bound CLIP/UNet/VAE graphs are installed";
    return false;
}
void SDXLRuntime::unload(){weights_=nullptr;}

} // namespace localimage::sdxl
