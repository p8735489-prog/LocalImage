#include "safetensors_to_limodel.h"
#include "../runtime/model_hash.h"
namespace localimage::converter {
bool SafeTensorsConverter::convert(const safetensors::SafeTensorFile&f,const std::string&p,std::string&e)const{models::Detection d=models::ModelDetector().detect(f);if(d.architecture==models::Architecture::Unknown){e="cannot convert: architecture is unknown";return false;}limodel::Manifest m;m.architecture=models::architectureName(d.architecture);m.tokenizer=d.components.clip||d.components.openclip;m.clip_l=d.components.clip;m.openclip_g=d.components.openclip;m.unet=d.components.unet;m.vae=d.components.vae;m.backends={"cpu"};
    localimage::runtime::ModelHash h;
    const auto* mapped=f.mappedData();
    if(!mapped || !h.compute(mapped,f.fileSize(),e)) return false;
    m.model_hash=h.hex();
    if(d.components.unet) m.backends.push_back("vulkan");
    return limodel::Writer().convert(f,p,m,e);}
}
