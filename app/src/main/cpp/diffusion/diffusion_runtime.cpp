#include "diffusion_runtime.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace localimage::diffusion {
using tensor::Tensor; using tensor::TensorDType; using tensor::TensorRuntime; using tensor::TensorShape;
namespace {
bool f32(const Tensor& t, std::string& e) { if (!t.valid() || t.dtype()!=TensorDType::F32 || !t.isContiguous()) { e="operator requires contiguous F32 tensor"; return false; } return true; }
bool sameShape(const Tensor&a,const Tensor&b){return a.shape().dims()==b.shape().dims();}
bool allocLike(const Tensor& x, Tensor& o, std::string&e){TensorRuntime rt;o=rt.createTensor(x.shape(),TensorDType::F32,e);return o.valid();}
}

bool DDIMScheduler::configure(const SchedulerConfig& c, std::string& e){
 if(c.inference_steps==0||c.inference_steps>10000||c.beta_start<=0||c.beta_end<=c.beta_start||c.beta_end>=1){e="invalid DDIM scheduler configuration";return false;}
 timesteps_.resize(c.inference_steps); alpha_bars_.resize(1000);
 double ab=1.0; for(size_t i=0;i<1000;i++){double f=(double)i/999.0;double beta=c.beta_start+(c.beta_end-c.beta_start)*f;ab*=1.0-beta;alpha_bars_[i]=ab;}
 for(size_t i=0;i<c.inference_steps;i++) timesteps_[i] = c.inference_steps == 1 ? 999.0 : 999.0 - (999.0 * (double)i) / (double)(c.inference_steps - 1);
 return true;
}
bool DDIMScheduler::step(const Tensor& mo,double t,const Tensor& x,Tensor&o,std::string&e)const{
 if(!f32(mo,e)||!f32(x,e)||!sameShape(mo,x)){e="DDIM step requires matching F32 tensors";return false;} if(alpha_bars_.empty()){e="DDIM scheduler is not configured";return false;}
 double tc=std::max(0.0,std::min(999.0,t)); size_t ti=(size_t)std::llround(tc); size_t pi=ti==0?0:ti-1; double a=alpha_bars_[ti], ap=alpha_bars_[pi]; double sa=std::sqrt(a), so=std::sqrt(std::max(1e-12,1.0-a)); double sap=std::sqrt(ap), sop=std::sqrt(std::max(1e-12,1.0-ap));
 if(!allocLike(x,o,e))return false;
 auto*O=(float*)o.mutableData();auto*E=(const float*)mo.data();auto*X=(const float*)x.data(); for(uint64_t i=0;i<x.shape().elementCount();i++){double pred=(X[i]-so*E[i])/sa;double y=sap*pred+sop*E[i];if(!std::isfinite(y)){e="DDIM produced non-finite output";return false;}O[i]=(float)y;} return true;
}

bool EulerScheduler::configure(const SchedulerConfig& c,std::string&e){
 if(c.inference_steps==0||c.inference_steps>10000){e="invalid Euler inference step count";return false;} timesteps_.resize(c.inference_steps);sigmas_.resize(c.inference_steps+1); for(size_t i=0;i<c.inference_steps;i++){double u=(double)i/(double)c.inference_steps;timesteps_[i]=1.0-u;sigmas_[i]=std::sqrt(std::max(0.0,(1.0-u)/(std::max(1e-12,u+1e-6))));}sigmas_[c.inference_steps]=0.0;return true;
}
bool EulerScheduler::step(const Tensor& mo,double t,const Tensor& x,Tensor&o,std::string&e)const{
 if(!f32(mo,e)||!f32(x,e)||!sameShape(mo,x)){e="Euler step requires matching F32 tensors";return false;} if(sigmas_.empty()){e="Euler scheduler is not configured";return false;} size_t i=0; double best=std::numeric_limits<double>::max();for(size_t j=0;j<timesteps_.size();j++){double d=std::fabs(t-timesteps_[j]);if(d<best){best=d;i=j;}} double s=sigmas_[i], sn=sigmas_[i+1], dt=sn-s; if(!allocLike(x,o,e))return false;
 auto*O=(float*)o.mutableData();auto*E=(const float*)mo.data();auto*X=(const float*)x.data();double denom=std::sqrt(1+s*s);for(uint64_t k=0;k<x.shape().elementCount();k++){double denoised=X[k]-s*E[k];double deriv=(X[k]-denoised)/std::max(1e-12,s);O[k]=(float)(X[k]+deriv*dt/denom);}return true;
}

bool FlowMatchScheduler::configure(const SchedulerConfig& c,std::string&e){if(c.inference_steps==0||c.inference_steps>10000){e="invalid FlowMatch inference step count";return false;}timesteps_.resize(c.inference_steps);for(size_t i=0;i<c.inference_steps;i++)timesteps_[i]=1.0-(double)i/(double)c.inference_steps;return true;}
bool FlowMatchScheduler::step(const Tensor& mo,double t,const Tensor& x,Tensor&o,std::string&e)const{if(!f32(mo,e)||!f32(x,e)||!sameShape(mo,x)){e="FlowMatch step requires matching F32 tensors";return false;}if(timesteps_.empty()){e="FlowMatch scheduler is not configured";return false;}double next=0;double best=std::numeric_limits<double>::max();for(size_t i=0;i<timesteps_.size();i++){double d=std::fabs(t-timesteps_[i]);if(d<best){best=d;next=(i+1<timesteps_.size()?timesteps_[i+1]:0.0);}}double dt=next-t;if(!allocLike(x,o,e))return false;
 auto*O=(float*)o.mutableData();auto*V=(const float*)mo.data();auto*X=(const float*)x.data();for(uint64_t i=0;i<x.shape().elementCount();i++)O[i]=(float)(X[i]+dt*V[i]);return true;}


bool DiffusionRuntime::inspect(const safetensors::SafeTensorFile& f,DiffusionModelInfo&o,std::string&e)const{
 if(f.tensors().empty()){e="model contains no tensors";return false;}o={};o.tensor_count=f.tensors().size();for(const auto& kv:f.tensors()){const auto& n=kv.first;o.parameter_bytes+=kv.second.byte_size;
  if(n.find("model.diffusion_model.")==0||n.find("diffusion_model.")==0||n.find("unet.")==0)o.components.unet=true;
  if(n.find("first_stage_model.")==0||n.find("vae.")==0)o.components.vae=true;
  if(n.find("cond_stage_model.transformer.text_model.")==0||n.find("text_model.encoder.")==0)o.components.clip=true;
  if(n.find("conditioner.embedders.1.")==0||n.find("text_encoder_2.")==0)o.components.openclip=true;
  if(n.find("text_encoder_3.")==0||n.find("t5xxl.")==0||n.find("t5.")==0)o.components.t5=true;
  if(n.find("tokenizer.")==0)o.components.tokenizer=true;
  if(n.find("text_projection")==0||n.find("text_projection.")!=std::string::npos)o.components.textProjection=true;
 }
 if(o.components.t5 && o.components.unet)o.architecture="SD3-family";
 else if(o.components.openclip && o.components.unet && o.components.clip)o.architecture="SDXL";
 else if(o.components.unet && o.components.clip)o.architecture="StableDiffusion-1/2";
 else if(o.components.t5 && !o.components.unet)o.architecture="Transformer-diffusion-family";
 else o.architecture="Unknown";
 return true;
}
bool DiffusionRuntime::createScheduler(SchedulerType t,const SchedulerConfig& c,std::unique_ptr<Scheduler>&o,std::string&e)const{std::unique_ptr<Scheduler> s;if(t==SchedulerType::DDIM)s=std::make_unique<DDIMScheduler>();
 else if(t==SchedulerType::Euler)s=std::make_unique<EulerScheduler>();
 else if(t==SchedulerType::FlowMatch)s=std::make_unique<FlowMatchScheduler>();
 else if(t==SchedulerType::EulerAncestral||t==SchedulerType::DPMPlusPlus2M){e="requested scheduler is unavailable; refusing to substitute Euler";return false;}
 else{e="unknown scheduler type";return false;}if(!s->configure(c,e))return false;o=std::move(s);return true;}

bool rmsNorm(const Tensor&x,const Tensor&w,double eps,Tensor&o,std::string&e){if(!f32(x,e)||!f32(w,e)||w.shape().rank()!=1||x.shape().rank()==0||w.shape().dim(0)!=x.shape().dim(x.shape().rank()-1)){e="RMSNorm shape mismatch";return false;}if(eps<=0){e="RMSNorm epsilon must be positive";return false;}if(!allocLike(x,o,e))return false;size_t n=w.shape().dim(0),rows=x.shape().elementCount()/n;auto*O=(float*)o.mutableData();auto*X=(const float*)x.data();auto*W=(const float*)w.data();for(size_t r=0;r<rows;r++){double ss=0;for(size_t j=0;j<n;j++){double v=X[r*n+j];ss+=v*v;}double inv=1.0/std::sqrt(ss/n+eps);for(size_t j=0;j<n;j++)O[r*n+j]=(float)(X[r*n+j]*inv*W[j]);}return true;}

bool rotaryEmbedding(const Tensor& x, const Tensor& freq, double scale, Tensor& o, std::string& e) {
 if(!f32(x,e)||!f32(freq,e)||x.shape().rank()!=3||freq.shape().rank()!=2){e="RoPE expects X[B,S,D] and frequencies[S,D/2]";return false;}
 const size_t B=x.shape().dim(0), S=x.shape().dim(1), D=x.shape().dim(2), H=D/2;
 if(freq.shape().dim(0)!=S||freq.shape().dim(1)>H){e="RoPE frequency shape must be [S,D/2] or a prefix of it";return false;}
 if(!allocLike(x,o,e))return false;
 const size_t FH=freq.shape().dim(1);auto*O=(float*)o.mutableData();auto*X=(const float*)x.data();auto*F=(const float*)freq.data();
 for(size_t b=0;b<B;b++) for(size_t s=0;s<S;s++) for(size_t d=0;d<D;d+=2){
   const size_t h=d/2; if(h>=FH){O[(b*S+s)*D+d]=X[(b*S+s)*D+d];if(d+1<D)O[(b*S+s)*D+d+1]=X[(b*S+s)*D+d+1];continue;}
   const double a=(double)F[s*FH+h]*scale,c=std::cos(a),sn=std::sin(a);const double q=X[(b*S+s)*D+d],r=d+1<D?X[(b*S+s)*D+d+1]:0;
   O[(b*S+s)*D+d]=(float)(q*c-r*sn);if(d+1<D)O[(b*S+s)*D+d+1]=(float)(q*sn+r*c);
 }
 return true;
}

bool timestepEmbedding(const Tensor&t,size_t dim,double maxPeriod,Tensor&o,std::string&e){if(!f32(t,e)||t.shape().rank()!=1||dim==0||dim%2!=0||maxPeriod<=1){e="timestepEmbedding requires 1D F32 input, positive even dimension and maxPeriod > 1";return false;}TensorRuntime rt;o=rt.createTensor(TensorShape({t.shape().dim(0),dim}),TensorDType::F32,e);if(!o.valid())return false;auto*O=(float*)o.mutableData();auto*T=(const float*)t.data();size_t half=dim/2;for(size_t n=0;n<t.shape().dim(0);n++)for(size_t i=0;i<half;i++){double f=std::exp(-std::log(maxPeriod)*(double)i/(double)half);double a=T[n]*f;O[n*dim+i]=(float)std::cos(a);O[n*dim+half+i]=(float)std::sin(a);}return true;}
}
