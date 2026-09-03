#include "sd15_runtime.h"
namespace localimage::sd15 {
bool SD15Runtime::inspect(const safetensors::SafeTensorFile& f,diffusion::DiffusionModelInfo&o,std::string&e)const{diffusion::DiffusionRuntime r;if(!r.inspect(f,o,e))return false;if(o.architecture!="StableDiffusion-1/2"){e="SD1.x runtime received a model outside the SD1.x/SD2.x UNet family";return false;}if(!o.components.unet||!o.components.vae||!o.components.clip){e="SD1.x model is missing required UNet, VAE, or CLIP weights";return false;}return true;}
bool SD15Runtime::createScheduler(diffusion::SchedulerType t,const diffusion::SchedulerConfig&c,std::unique_ptr<diffusion::Scheduler>&o,std::string&e)const{diffusion::DiffusionRuntime r;return r.createScheduler(t,c,o,e);}
}
