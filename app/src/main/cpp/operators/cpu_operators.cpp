#include "cpu_operators.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

using localimage::tensor::Tensor;
using localimage::tensor::TensorDType;
using localimage::tensor::TensorRuntime;
using localimage::tensor::TensorShape;

namespace localimage::ops {
bool add(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return tensor::ops::add(a,b,o,e);}
bool sub(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return tensor::ops::sub(a,b,o,e);}
bool mul(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return tensor::ops::mul(a,b,o,e);}
bool div(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return tensor::ops::div(a,b,o,e);}

namespace {
bool f32(const Tensor& t, std::string& e) {
 if (!t.valid()) { e="CPU operator received invalid tensor"; return false; }
 if (t.dtype()!=TensorDType::F32) { e="CPU operator currently requires F32 tensor"; return false; }
 if (!t.isContiguous()) { e="CPU operator requires contiguous tensor"; return false; }
 return true;
}

bool makeLike(const TensorShape& s, Tensor& out, std::string& e){ TensorRuntime rt; out=rt.createTensor(s,TensorDType::F32,e); return out.valid(); }
uint64_t product(const std::vector<uint64_t>& d,size_t a,size_t b){uint64_t n=1;for(size_t i=a;i<b;i++)n*=d[i];return n;}
}

bool unary(const Tensor& x, Tensor& out, const std::string& op, std::string& e){
 if(!f32(x,e))return false;
 if(op!="neg"&&op!="exp"&&op!="sqrt"&&op!="rsqrt"&&op!="clamp"&&op!="silu"&&op!="gelu"){e="unsupported unary op: "+op;return false;}
 if(!makeLike(x.shape(),out,e))return false;
 const float* p=(const float*)x.data(); float* q=(float*)out.mutableData();
 for(uint64_t i=0;i<x.shape().elementCount();++i){
  float v=p[i];
  if(op=="neg") q[i]=-v;
  else if(op=="exp") q[i]=std::exp(v);
  else if(op=="sqrt"){ if(v<0){e="sqrt received a negative value";return false;} q[i]=std::sqrt(v); }
  else if(op=="rsqrt"){ if(v<=0){e="rsqrt requires positive input";return false;} q[i]=1.0f/std::sqrt(v); }
  else if(op=="clamp") q[i]=std::max(-1.0f,std::min(1.0f,v));
  else if(op=="silu") q[i]=v/(1.0f+std::exp(-v));
  else q[i]=0.5f*v*(1.0f+std::erf(v/std::sqrt(2.0f)));
  if(!std::isfinite(q[i])){e="unary operator produced non-finite output";return false;}
 }
 return true;
}

bool reduce(const Tensor& x,const std::vector<size_t>& axes,bool keepdim,Tensor& out,const std::string& op,std::string&e){
 if(!f32(x,e)) return false;
 std::unordered_set<size_t> ax;
 for(auto a:axes){ if(a>=x.shape().rank()){e="reduction axis out of range";return false;} ax.insert(a); }
 if(ax.empty()){e="reduction requires at least one axis";return false;}
 std::vector<uint64_t> od;
 for(size_t i=0;i<x.shape().rank();++i) if(ax.count(i)){if(keepdim) od.push_back(1);} else od.push_back(x.shape().dim(i));
 if(!makeLike(TensorShape(od),out,e)) return false;
 float* dst=(float*)out.mutableData();
 std::fill(dst,dst+out.shape().elementCount(),op=="max"?-std::numeric_limits<float>::infinity():0.0f);
 const float* src=(const float*)x.data(); uint64_t inN=x.shape().elementCount(); std::vector<uint64_t> idx(x.shape().rank());
 for(uint64_t flat=0;flat<inN;++flat){uint64_t r=flat;for(size_t d=x.shape().rank();d-->0;){idx[d]=r%x.shape().dim(d);r/=x.shape().dim(d);}   std::vector<uint64_t> odidx;for(size_t d=0;d<idx.size();++d)if(!ax.count(d))odidx.push_back(idx[d]);else if(keepdim)odidx.push_back(0);uint64_t of=0;for(size_t d=0;d<odidx.size();++d){uint64_t stride=1;for(size_t k=d+1;k<odidx.size();++k)stride*=out.shape().dim(k);of+=odidx[d]*stride;}
   if(op=="sum"||op=="mean")dst[of]+=src[flat];else if(op=="max")dst[of]=std::max(dst[of],src[flat]);
 }
 if(op=="mean"){uint64_t denom=1;for(auto a:ax)denom*=x.shape().dim(a);for(uint64_t i=0;i<out.shape().elementCount();++i)dst[i]/=(float)denom;}
 return true;
}

bool matmul(const Tensor&a,const Tensor&b,Tensor&out,std::string&e){
 if(!f32(a,e)||!f32(b,e))return false;
 if(a.shape().rank()!=2||b.shape().rank()!=2){e="matmul requires rank-2 tensors [M,K] x [K,N]";return false;}
 const size_t M=a.shape().dim(0),K=a.shape().dim(1),N=b.shape().dim(1);
 if(K!=b.shape().dim(0)){e="matmul K dimension mismatch";return false;}
 if(M && N > UINT64_MAX/M){e="matmul output shape overflow";return false;}
 if(!makeLike(TensorShape({M,N}),out,e))return false;
 auto*A=(const float*)a.data();auto*B=(const float*)b.data();auto*C=(float*)out.mutableData();
 for(size_t i=0;i<M;i++)for(size_t j=0;j<N;j++){
  double sum=0; for(size_t k=0;k<K;k++) sum+=(double)A[i*K+k]*B[k*N+j];
  if(!std::isfinite(sum)){e="matmul produced non-finite output";return false;}
  C[i*N+j]=(float)sum;
 }
 return true;
}

bool batchedMatmul(const Tensor&a,const Tensor&b,Tensor&out,std::string&e){
 if(!f32(a,e)||!f32(b,e))return false;
 const size_t ar=a.shape().rank(), br=b.shape().rank();
 if(ar<2||br<2||ar!=br){e="BatchedMatMul requires equal rank >= 2";return false;}
 const size_t batchRank=ar-2;
 for(size_t i=0;i<batchRank;i++){uint64_t ad=a.shape().dim(i),bd=b.shape().dim(i);if(ad!=bd&&ad!=1&&bd!=1){e="BatchedMatMul batch dimensions are not broadcastable";return false;}}
 const uint64_t M=a.shape().dim(ar-2),K=a.shape().dim(ar-1),BK=b.shape().dim(br-2),N=b.shape().dim(br-1);
 if(K!=BK){e="BatchedMatMul K dimension mismatch";return false;}
 std::vector<uint64_t> od(batchRank);for(size_t i=0;i<batchRank;i++)od[i]=std::max(a.shape().dim(i),b.shape().dim(i));od.push_back(M);od.push_back(N);TensorShape shape(std::move(od));if(!shape.valid()){e=shape.error();return false;}
 TensorRuntime rt;out=rt.createTensor(shape,TensorDType::F32,e);if(!out.valid())return false;
 const float*A=(const float*)a.data(),*B=(const float*)b.data();float*O=(float*)out.mutableData();
 const uint64_t aMat=M*K,bMat=K*N,oMat=M*N,batches=shape.elementCount()/oMat;
 std::vector<uint64_t> bi(batchRank);
 for(uint64_t q=0;q<batches;q++){uint64_t rem=q;for(size_t d=batchRank;d-->0;){bi[d]=shape.dim(d)?rem%shape.dim(d):0;rem=shape.dim(d)?rem/shape.dim(d):0;}uint64_t ab=0,bb=0;for(size_t d=0;d<batchRank;d++){uint64_t ai=a.shape().dim(d)==1?0:bi[d], bj=b.shape().dim(d)==1?0:bi[d];uint64_t as=1,bs=1;for(size_t k=d+1;k<batchRank;k++){as*=a.shape().dim(k);bs*=b.shape().dim(k);}ab+=ai*as;bb+=bj*bs;}for(size_t i=0;i<M;i++)for(size_t j=0;j<N;j++){double sum=0;for(size_t k=0;k<K;k++)sum+=(double)A[ab*aMat+i*K+k]*B[bb*bMat+k*N+j];if(!std::isfinite(sum)){e="BatchedMatMul produced non-finite output";return false;}O[q*oMat+i*N+j]=(float)sum;}}
 return true;
}

bool linear(const Tensor&x,const Tensor&w,const Tensor*bias,Tensor&out,std::string&e){if(!matmul(x,w,out,e))return false;if(bias){if(!f32(*bias,e)||bias->shape().rank()!=1||bias->shape().dim(0)!=out.shape().dim(out.shape().rank()-1)){e="linear bias shape mismatch";return false;}float*o=(float*)out.mutableData();auto*b=(const float*)bias->data();size_t n=bias->shape().dim(0);for(uint64_t i=0;i<out.shape().elementCount();++i)o[i]+=b[i%n];}return true;}

bool layerNorm(const Tensor&x,const Tensor&g,const Tensor*b,double eps,Tensor&out,std::string&e){if(!f32(x,e)||!f32(g,e)||(b&&!f32(*b,e)))return false;if(x.shape().rank()==0||g.shape().rank()!=1||g.shape().dim(0)!=x.shape().dim(x.shape().rank()-1)){e="layernorm normalized shape mismatch";return false;}if(b&&b->shape().dims()!=g.shape().dims()){e="layernorm beta shape mismatch";return false;}if(!makeLike(x.shape(),out,e))return false;size_t N=g.shape().dim(0);uint64_t rows=x.shape().elementCount()/N;auto*X=(const float*)x.data();auto*G=(const float*)g.data();auto*B=b?(const float*)b->data():nullptr;auto*O=(float*)out.mutableData();for(uint64_t r=0;r<rows;r++){double m=0;for(size_t j=0;j<N;j++)m+=X[r*N+j];m/=N;double v=0;for(size_t j=0;j<N;j++){double z=X[r*N+j]-m;v+=z*z;}v/=N;double inv=1.0/std::sqrt(v+eps);for(size_t j=0;j<N;j++)O[r*N+j]=(float)((X[r*N+j]-m)*inv*G[j]+(B?B[j]:0));}return true;}

bool groupNorm(const Tensor&x,const Tensor&g,const Tensor*b,size_t groups,double eps,Tensor&out,std::string&e){if(!f32(x,e)||!f32(g,e)||(b&&!f32(*b,e)))return false;if(x.shape().rank()!=4){e="GroupNorm requires NCHW";return false;}size_t N=x.shape().dim(0),C=x.shape().dim(1),H=x.shape().dim(2),W=x.shape().dim(3);if(groups==0||C%groups){e="channels must be divisible by groups";return false;}if(g.shape().rank()!=1||g.shape().dim(0)!=C||(b&&b->shape().dims()!=g.shape().dims())){e="GroupNorm gamma/beta shape mismatch";return false;}if(!makeLike(x.shape(),out,e))return false;auto*X=(const float*)x.data();auto*G=(const float*)g.data();auto*B=b?(const float*)b->data():nullptr;auto*O=(float*)out.mutableData();size_t per=C/groups*H*W;for(size_t n=0;n<N;n++)for(size_t gr=0;gr<groups;gr++){size_t base=((n*C)+gr*(C/groups))*H*W;double m=0;for(size_t i=0;i<per;i++)m+=X[base+i];m/=per;double v=0;for(size_t i=0;i<per;i++){double z=X[base+i]-m;v+=z*z;}v/=per;double inv=1/std::sqrt(v+eps);for(size_t c=0;c<C/groups;c++)for(size_t s=0;s<H*W;s++){size_t i=base+c*H*W+s;O[i]=(float)((X[i]-m)*inv*G[gr*(C/groups)+c]+(B?B[gr*(C/groups)+c]:0));}}return true;}

bool conv2d(const Tensor&x,const Tensor&w,const Tensor*b,size_t st,size_t pad,size_t dil,size_t groups,Tensor&out,std::string&e){if(!f32(x,e)||!f32(w,e)||(b&&!f32(*b,e)))return false;if(x.shape().rank()!=4||w.shape().rank()!=4||groups==0){e="Conv2D requires NCHW 4D tensors";return false;}size_t N=x.shape().dim(0),Cin=x.shape().dim(1),H=x.shape().dim(2),W=x.shape().dim(3),Cout=w.shape().dim(0),Kc=w.shape().dim(1),KH=w.shape().dim(2),KW=w.shape().dim(3);if(Cin%groups||Cout%groups||Kc!=Cin/groups){e="unsupported Conv2D groups/channel layout";return false;}if(st==0||dil==0){e="stride/dilation must be > 0";return false;}if(KH==0||KW==0){e="Conv2D kernel dimensions must be non-zero";return false;}
 uint64_t effH=1+(uint64_t)dil*(KH-1), effW=1+(uint64_t)dil*(KW-1);
 uint64_t numH=(uint64_t)H+2ULL*pad, numW=(uint64_t)W+2ULL*pad;
 if(numH<effH||numW<effW){e="Conv2D kernel is larger than padded input";return false;}
 size_t OH=(size_t)((numH-effH)/st+1),OW=(size_t)((numW-effW)/st+1);if(!makeLike(TensorShape({N,Cout,OH,OW}),out,e))return false;auto*X=(const float*)x.data();auto*Wg=(const float*)w.data();auto*B=b?(const float*)b->data():nullptr;auto*O=(float*)out.mutableData();std::fill(O,O+out.shape().elementCount(),0);for(size_t n=0;n<N;n++)for(size_t co=0;co<Cout;co++)for(size_t oh=0;oh<OH;oh++)for(size_t ow=0;ow<OW;ow++){double s=B?B[co]:0;size_t gi=co/(Cout/groups),cin0=gi*Kc;for(size_t ci=0;ci<Kc;ci++)for(size_t kh=0;kh<KH;kh++)for(size_t kw=0;kw<KW;kw++){long ih=(long)(oh*st+kh*dil)-pad,iw=(long)(ow*st+kw*dil)-pad;if(ih<0||iw<0||ih>=(long)H||iw>=(long)W)continue;size_t ic=cin0+ci;s+=(double)X[((n*Cin+ic)*H+ih)*W+iw]*Wg[((co*Kc+ci)*KH+kh)*KW+kw];}O[((n*Cout+co)*OH+oh)*OW+ow]=(float)s;}return true;}

bool upsampleNearest(const Tensor&x,size_t scale,Tensor&out,std::string&e){if(!f32(x,e)||x.shape().rank()!=4||scale==0){e="nearest upsample requires NCHW and scale > 0";return false;}auto d=x.shape().dims();d[2]*=scale;d[3]*=scale;if(!makeLike(TensorShape(d),out,e))return false;auto*X=(const float*)x.data();auto*O=(float*)out.mutableData();size_t N=x.shape().dim(0),C=x.shape().dim(1),H=x.shape().dim(2),W=x.shape().dim(3),OH=H*scale,OW=W*scale;for(size_t n=0;n<N;n++)for(size_t c=0;c<C;c++)for(size_t h=0;h<OH;h++)for(size_t w=0;w<OW;w++)O[((n*C+c)*OH+h)*OW+w]=X[((n*C+c)*H+h/scale)*W+w/scale];return true;}

bool concat(const std::vector<Tensor>&xs,size_t axis,Tensor&out,std::string&e){if(xs.empty()){e="concat requires tensors";return false;}for(const auto&x:xs)if(!f32(x,e))return false;size_t r=xs[0].shape().rank();if(axis>=r){e="concat axis out of range";return false;}auto d=xs[0].shape().dims();uint64_t sum=0;for(const auto&x:xs){if(x.shape().rank()!=r){e="concat rank mismatch";return false;}for(size_t i=0;i<r;i++)if(i!=axis&&x.shape().dim(i)!=d[i]){e="concat shape mismatch";return false;}if(sum>UINT64_MAX-x.shape().dim(axis)){e="concat dimension overflow";return false;}sum+=x.shape().dim(axis);}d[axis]=sum;if(!makeLike(TensorShape(d),out,e))return false;size_t inner=1;for(size_t i=axis+1;i<r;i++)inner*=d[i];size_t outer=product(d,0,axis);float*O=(float*)out.mutableData();size_t dst=0;for(size_t o=0;o<outer;o++)for(const auto&x:xs){size_t len=x.shape().dim(axis)*inner;const float*X=(const float*)x.data()+o*len;std::copy(X,X+len,O+dst);dst+=len;}return true;}

bool transpose(const Tensor&x,const std::vector<size_t>&perm,Tensor&out,std::string&e){if(!f32(x,e))return false;size_t r=x.shape().rank();if(perm.size()!=r){e="transpose permutation rank mismatch";return false;}std::vector<bool>seen(r);for(auto p:perm){if(p>=r||seen[p]){e="invalid transpose permutation";return false;}seen[p]=true;}std::vector<uint64_t>d(r);for(size_t i=0;i<r;i++)d[i]=x.shape().dim(perm[i]);if(!makeLike(TensorShape(d),out,e))return false;auto*X=(const float*)x.data();auto*O=(float*)out.mutableData();for(uint64_t of=0;of<out.shape().elementCount();of++){uint64_t q=of,xf=0;std::vector<uint64_t> oi(r);for(size_t i=r;i-->0;){oi[i]=q%out.shape().dim(i);q/=out.shape().dim(i);}std::vector<uint64_t> xi(r);for(size_t i=0;i<r;i++)xi[perm[i]]=oi[i];for(size_t i=0;i<r;i++){uint64_t s=1;for(size_t k=i+1;k<r;k++)s*=x.shape().dim(k);xf+=xi[i]*s;}O[of]=X[xf];}return true;}

bool softmax(const Tensor&x,size_t axis,Tensor&out,std::string&e){if(!f32(x,e))return false;if(axis>=x.shape().rank()){e="softmax axis out of range";return false;}if(!makeLike(x.shape(),out,e))return false;size_t inner=1;for(size_t i=axis+1;i<x.shape().rank();i++)inner*=x.shape().dim(i);size_t axisN=x.shape().dim(axis),outer=x.shape().elementCount()/(axisN*inner);auto*X=(const float*)x.data();auto*O=(float*)out.mutableData();for(size_t o=0;o<outer;o++)for(size_t in=0;in<inner;in++){float mx=-std::numeric_limits<float>::infinity();for(size_t a=0;a<axisN;a++)mx=std::max(mx,X[(o*axisN+a)*inner+in]);double sum=0;for(size_t a=0;a<axisN;a++){double e2=std::exp((double)X[(o*axisN+a)*inner+in]-mx);O[(o*axisN+a)*inner+in]=(float)e2;sum+=e2;}for(size_t a=0;a<axisN;a++)O[(o*axisN+a)*inner+in]=(float)(O[(o*axisN+a)*inner+in]/sum);}return true;}

bool scaledDotProductAttention(const Tensor&q,const Tensor&k,const Tensor&v,double scale,const Tensor*mask,Tensor&out,std::string&e){
 if(!f32(q,e)||!f32(k,e)||!f32(v,e)||(mask&&!f32(*mask,e)))return false;
 const size_t qr=q.shape().rank(), kr=k.shape().rank(), vr=v.shape().rank();
 if(qr!=kr||kr!=vr||(qr!=3&&qr!=4)){e="Attention expects rank-3 [B,S,D] or rank-4 [B,H,S,D] tensors";return false;}
 if(scale<=0.0||!std::isfinite(scale)){e="Attention scale must be finite and > 0";return false;}
 const size_t B=q.shape().dim(0), H=(qr==4?q.shape().dim(1):1), Q=(qr==4?q.shape().dim(2):q.shape().dim(1));
 const size_t K=(qr==4?k.shape().dim(2):k.shape().dim(1)), D=(qr==4?q.shape().dim(3):q.shape().dim(2)), Vd=(qr==4?v.shape().dim(3):v.shape().dim(2));
 if(k.shape().dim(0)!=B||v.shape().dim(0)!=B||q.shape().dim(qr-1)!=k.shape().dim(kr-1)||k.shape().dim(kr-2)!=v.shape().dim(vr-2)) {e="Attention Q/K/V shape mismatch";return false;}
 if(qr==4 && (k.shape().dim(1)!=H||v.shape().dim(1)!=H)){e="Attention head count mismatch";return false;}
 TensorShape os(qr==4?std::vector<uint64_t>{B,H,Q,Vd}:std::vector<uint64_t>{B,Q,Vd});
 TensorRuntime rt;out=rt.createTensor(os,TensorDType::F32,e);if(!out.valid())return false;
 const float*Qp=static_cast<const float*>(q.data()),*Kp=static_cast<const float*>(k.data()),*Vp=static_cast<const float*>(v.data()),*Mp=mask?static_cast<const float*>(mask->data()):nullptr;float*O=static_cast<float*>(out.mutableData());
 std::vector<double> scores(K);
 for(size_t b=0;b<B;++b) for(size_t h=0;h<H;++h) for(size_t qi=0;qi<Q;++qi){
   double mx=-std::numeric_limits<double>::infinity();
   for(size_t ki=0;ki<K;++ki){double z=0;for(size_t d=0;d<D;++d){const size_t qo=qr==4?(((b*H+h)*Q+qi)*D+d):((b*Q+qi)*D+d);const size_t ko=qr==4?(((b*H+h)*K+ki)*D+d):((b*K+ki)*D+d);z+=(double)Qp[qo]*Kp[ko];}scores[ki]=z*scale;
     if(Mp){const auto md=mask->shape().dims();size_t mo=0;if(md.size()==4&&md[0]==B&&md[1]==H&&md[2]==Q&&md[3]==K)mo=(((b*H+h)*Q+qi)*K+ki);else if(md.size()==3&&md[0]==B&&md[1]==Q&&md[2]==K)mo=((b*Q+qi)*K+ki);else if(md.size()==2&&md[0]==Q&&md[1]==K)mo=qi*K+ki;else {e="Attention mask must be [B,H,Q,K], [B,Q,K] or [Q,K]";return false;}scores[ki]+=Mp[mo];}
     mx=std::max(mx,scores[ki]);
   }
   double sum=0;for(auto&z:scores){z=std::exp(z-mx);sum+=z;}if(!(sum>0)||!std::isfinite(sum)){e="Attention softmax produced invalid normalization";return false;}
   for(size_t vi=0;vi<Vd;++vi){double z=0;for(size_t ki=0;ki<K;++ki){const size_t vo=qr==4?(((b*H+h)*K+ki)*Vd+vi):((b*K+ki)*Vd+vi);z+=(scores[ki]/sum)*Vp[vo];}const size_t oo=qr==4?(((b*H+h)*Q+qi)*Vd+vi):((b*Q+qi)*Vd+vi);O[oo]=(float)z;}
 }
 return true;
}

bool rmsNorm(const Tensor& x, const Tensor& gamma, double eps, Tensor& out, std::string& e) {
    if (!f32(x,e) || !f32(gamma,e)) return false;
    if (eps <= 0.0 || x.shape().rank() == 0 || gamma.shape().rank() != 1 ||
        gamma.shape().dim(0) != x.shape().dim(x.shape().rank()-1)) {
        e = "RMSNorm shape or epsilon mismatch"; return false;
    }
    if (!makeLike(x.shape(),out,e)) return false;
    const size_t n=gamma.shape().dim(0); const uint64_t rows=x.shape().elementCount()/n;
    const float* X=static_cast<const float*>(x.data()); const float* G=static_cast<const float*>(gamma.data());
    float* O=static_cast<float*>(out.mutableData());
    for(uint64_t r=0;r<rows;++r){
        double ss=0.0; for(size_t j=0;j<n;++j){double v=X[r*n+j];ss+=v*v;}
        const double inv=1.0/std::sqrt(ss/static_cast<double>(n)+eps);
        for(size_t j=0;j<n;++j) O[r*n+j]=static_cast<float>(X[r*n+j]*inv*G[j]);
    }
    return true;
}

bool rope(const Tensor& x, const Tensor& angles, Tensor& out, std::string& e) {
    if(!f32(x,e)||!f32(angles,e)) return false;
    if(x.shape().rank()!=3 || angles.shape().rank()!=2 ||
       angles.shape().dim(0)!=x.shape().dim(1) || angles.shape().dim(1)*2>x.shape().dim(2)) {
        e="RoPE expects X=[B,S,D] and angles=[S,D/2]"; return false;
    }
    if(!makeLike(x.shape(),out,e)) return false;
    const size_t B=x.shape().dim(0),S=x.shape().dim(1),D=x.shape().dim(2),H=angles.shape().dim(1);
    const float* X=static_cast<const float*>(x.data()); const float* A=static_cast<const float*>(angles.data()); float* O=static_cast<float*>(out.mutableData());
    for(size_t b=0;b<B;++b) for(size_t s0=0;s0<S;++s0) for(size_t d=0;d<D;d+=2){
        if(d/2>=H){ O[(b*S+s0)*D+d]=X[(b*S+s0)*D+d]; if(d+1<D) O[(b*S+s0)*D+d+1]=X[(b*S+s0)*D+d+1]; continue; }
        const double a=A[s0*H+d/2], c=std::cos(a), sn=std::sin(a);
        const double p=X[(b*S+s0)*D+d], q=(d+1<D?X[(b*S+s0)*D+d+1]:0.0);
        O[(b*S+s0)*D+d]=static_cast<float>(p*c-q*sn); if(d+1<D) O[(b*S+s0)*D+d+1]=static_cast<float>(p*sn+q*c);
    }
    return true;
}

bool broadcastTo(const Tensor& x, const TensorShape& shape, Tensor& out, std::string& e) {
    if(!f32(x,e) || !shape.valid()){ if(e.empty()) e="invalid broadcast shape"; return false; }
    TensorShape src=x.shape(); if(src.rank()>shape.rank()){e="broadcast target rank is smaller";return false;}
    const size_t off=shape.rank()-src.rank();
    for(size_t i=0;i<src.rank();++i) if(src.dim(i)!=1 && src.dim(i)!=shape.dim(off+i)){e="broadcast dimensions are incompatible";return false;}
    if(!makeLike(shape,out,e))return false;
    const float* X=static_cast<const float*>(x.data()); float* O=static_cast<float*>(out.mutableData());
    for(uint64_t linear=0;linear<shape.elementCount();++linear){
        uint64_t rem=linear, xo=0;
        for(size_t d=shape.rank();d-->0;){
            const uint64_t idx=shape.dim(d)?rem%shape.dim(d):0; rem=shape.dim(d)?rem/shape.dim(d):0;
            if(d>=off){ const size_t sd=d-off; const uint64_t si=src.dim(sd)==1?0:idx; uint64_t st=1; for(size_t k=sd+1;k<src.rank();++k) st*=src.dim(k); xo+=si*st; }
        }
        O[linear]=X[xo];
    }
    return true;
}

ErrorMetrics compare(const Tensor&a,const Tensor&b,std::string&e){ErrorMetrics m;if(!f32(a,e)||!f32(b,e))return m;if(a.shape().dims()!=b.shape().dims()){e="comparison shape mismatch";return m;}const float*A=(const float*)a.data(),*B=(const float*)b.data();uint64_t n=a.shape().elementCount();double sum=0;for(uint64_t i=0;i<n;i++){double aa=A[i],bb=B[i],abs=std::fabs(aa-bb),rel=abs/(std::fabs(bb)+1e-12);m.max_abs=std::max(m.max_abs,abs);m.max_rel=std::max(m.max_rel,rel);sum+=abs;}m.mean_abs=n?sum/n:0;return m;}
}
