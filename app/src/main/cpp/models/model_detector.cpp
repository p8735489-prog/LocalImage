#include "model_detector.h"
#include <algorithm>
#include <unordered_set>
namespace localimage::models {
const char* architectureName(Architecture a){switch(a){case Architecture::StableDiffusion15:return "Stable Diffusion 1.x";case Architecture::StableDiffusion2:return "Stable Diffusion 2.x";case Architecture::SDXL:return "SDXL";case Architecture::SD3:return "SD3";case Architecture::SD35:return "SD3.5";case Architecture::FLUX:return "FLUX";case Architecture::Anima:return "Anima";default:return "Unknown";}}
Detection ModelDetector::detect(const safetensors::SafeTensorFile& f)const{
 Detection d; if(f.tensors().empty()){d.reason="model contains no tensors";return d;}
 bool legacyUnet=false,modernUnet=false,sdxl=false,sd3=false,sd35=false,flux=false,anima=false,vae=false,clip=false,openclip=false,t5=false;
 for(const auto& kv:f.tensors()){const std::string& n=kv.first;
  legacyUnet|=n.find("model.diffusion_model.input_blocks.")==0||n.find("model.diffusion_model.output_blocks.")==0;
  modernUnet|=n.find("unet.")==0||n.find("diffusion_model.")==0;
  sdxl|=n.find("conditioner.embedders.0.")==0||n.find("conditioner.embedders.1.")==0||
       n.find("add_embedding.linear_1")==0||n.find("add_embedding.linear_2")==0||n.find("label_emb.0.0")==0;
  sd3|=n.find("joint_blocks.")!=std::string::npos||n.find("x_embedder.")!=std::string::npos;
  sd35|=n.find("context_embedder.")!=std::string::npos||n.find("y_embedder.")!=std::string::npos;
  flux|=n.find("double_blocks.")!=std::string::npos||n.find("single_blocks.")!=std::string::npos;
  anima|=n.find("anima")!=std::string::npos||n.find("Anima")!=std::string::npos;
  vae|=n.find("first_stage_model.")==0||n.find("vae.")==0||n.find("decoder.")==0;
  clip|=n.find("cond_stage_model.transformer.text_model.")==0||n.find("text_encoder.")!=std::string::npos||n.find("clip_l.")!=std::string::npos;
  openclip|=n.find("conditioner.embedders.1.")==0||n.find("text_encoder_2.")==0||n.find("clip_g.")!=std::string::npos||n.find("cond_stage_model.model.")==0||n.find("open_clip.")==0;
  t5|=n.find("text_encoder_3.")==0||n.find("t5xxl.")!=std::string::npos||n.find("t5.")!=std::string::npos;
 }
 d.components={legacyUnet||modernUnet,vae,clip,openclip,t5,sd3||sd35||flux};
 if(anima){d.architecture=Architecture::Anima;d.confidence=0.85;d.reason="Anima-like tensor metadata detected; runtime execution is not enabled";d.requiredComponents={"Anima transformer","text encoder","VAE","scheduler"};return d;}
 if(flux){d.architecture=Architecture::FLUX;d.confidence=(t5&&d.components.transformer)?0.99:0.90;d.supported=false;d.reason="FLUX architecture detected; transformer/text-encoder execution requires the diffusion transformer runtime";d.requiredComponents={"Flux transformer","CLIP-L","T5-XXL","VAE","FlowMatch scheduler"};return d;}
 if(sd35){d.architecture=Architecture::SD35;d.confidence=0.98;d.reason="SD3.5 tensor architecture detected";d.requiredComponents={"MMDiT","CLIP","T5","VAE","FlowMatch scheduler"};return d;}
 if(sd3){d.architecture=Architecture::SD3;d.confidence=0.98;d.reason="SD3 tensor architecture detected";d.requiredComponents={"MMDiT","CLIP","T5","VAE","FlowMatch scheduler"};return d;}
 if(sdxl&&clip&&openclip){d.architecture=Architecture::SDXL;d.confidence=0.98;d.reason="SDXL tensor architecture detected";d.requiredComponents={"UNet-XL","CLIP-L","OpenCLIP-G","VAE","scheduler"};return d;}
 if(legacyUnet&&clip){d.architecture=Architecture::StableDiffusion15;d.confidence=0.97;d.supported=false;d.reason="Stable Diffusion 1.x architecture detected; full model execution is not enabled in this runtime";d.requiredComponents={"UNet","CLIP","VAE","scheduler"};return d;}
 if(modernUnet&&(openclip||clip)){d.architecture=Architecture::StableDiffusion2;d.confidence=openclip?0.96:0.90;d.reason="Stable Diffusion 2.x-style UNet/text encoder namespaces detected";d.requiredComponents={"UNet","OpenCLIP","VAE","scheduler"};return d;}
 d.reason="model architecture could not be established from tensor metadata";return d;
}
}
