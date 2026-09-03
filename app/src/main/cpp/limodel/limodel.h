#pragma once
#include "../safetensors/safe_tensor_file.h"
#include "../runtime/ir/localimage_ir.h"
#include <string>
#include <vector>
namespace localimage::limodel {
struct Manifest { uint32_t version=1; std::string architecture; std::string dtype="F16"; std::string quantization="none"; std::string model_hash; bool tokenizer=false,clip_l=false,openclip_g=false,unet=false,vae=false; std::vector<std::string> backends; };
class Writer { public: bool convert(const safetensors::SafeTensorFile&,const std::string&,const Manifest&,std::string&) const; };
class Reader { public: bool readManifest(const std::string&,Manifest&,std::string&) const; bool valid(const std::string&,std::string&) const; };
std::string manifestJson(const Manifest&);
}
