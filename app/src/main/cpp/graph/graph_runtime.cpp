#include "graph_runtime.h"
#include "../operators/cpu_operators.h"
#ifndef LOCALIMAGE_NO_VULKAN
#include "../vulkan/vulkan_context.h"
#include "../vulkan/vulkan_compute.h"
#endif
#include "../runtime/ir/localimage_ir.h"
#include <queue>
#include <set>
#include <sstream>
#include <cmath>
#include <limits>

using namespace localimage::tensor;

namespace localimage::graph {

const char* opName(OpType o) {
 switch (o) {
  case OpType::Input:return "Input"; case OpType::Add:return "Add"; case OpType::Sub:return "Sub";
  case OpType::Mul:return "Mul"; case OpType::Div:return "Div"; case OpType::Exp:return "Exp"; case OpType::Sqrt:return "Sqrt"; case OpType::Rsqrt:return "Rsqrt"; case OpType::Clamp:return "Clamp"; case OpType::MatMul:return "MatMul";
  case OpType::Linear:return "Linear"; case OpType::BatchedMatMul:return "BatchedMatMul"; case OpType::Conv2D:return "Conv2D"; case OpType::GroupNorm:return "GroupNorm";
  case OpType::LayerNorm:return "LayerNorm"; case OpType::RMSNorm:return "RMSNorm"; case OpType::SiLU:return "SiLU"; case OpType::GELU:return "GELU";
  case OpType::Softmax:return "Softmax"; case OpType::Reshape:return "Reshape"; case OpType::Transpose:return "Transpose";
  case OpType::Concat:return "Concat"; case OpType::Slice:return "Slice"; case OpType::Broadcast:return "Broadcast"; case OpType::Upsample:return "Upsample"; case OpType::Attention:return "Attention"; case OpType::RoPE:return "RoPE";
 }
 return "Unknown";
}

uint32_t Graph::addValue(std::string n,const TensorShape& s,TensorDType d) {
 uint32_t id=(uint32_t)values_.size();
 values_.push_back({id,std::move(n),s,d});
 return id;
}

uint32_t Graph::addNode(OpType op,std::vector<uint32_t> in,std::string n) {
 uint32_t id=(uint32_t)nodes_.size();
 GraphNode node;
 node.id=id; node.op=op; node.inputs=std::move(in); node.name=std::move(n);
 nodes_.push_back(std::move(node));
 return id;
}

bool Graph::setNodeOutputs(uint32_t id,std::vector<uint32_t> o,std::string& e) {
 if(id>=nodes_.size()){e="graph node id out of range";return false;}
 if(o.empty()){e="graph node must have at least one output";return false;}
 std::set<uint32_t> seen;
 for(auto v:o) {
  if(v>=values_.size()){e="graph value id out of range";return false;}
  if(!seen.insert(v).second){e="graph node contains duplicate output value";return false;}
 }
 nodes_[id].outputs=std::move(o);
 return true;
}

bool Graph::setNodeAttributes(uint32_t id, std::vector<size_t> axes, std::vector<size_t> permutation,
                              size_t groups, size_t stride, size_t padding, size_t dilation,
                              size_t scale_factor, double epsilon, double attention_scale,
                              bool keepdim, std::string& e) {
 if(id>=nodes_.size()){e="graph node id out of range";return false;}
 if(stride==0||dilation==0||scale_factor==0||groups==0||epsilon<=0){e="invalid graph operator attributes";return false;}
 auto& n=nodes_[id];
 n.axes=std::move(axes); n.permutation=std::move(permutation); n.groups=groups;
 n.stride=stride; n.padding=padding; n.dilation=dilation; n.scale_factor=scale_factor;
 n.epsilon=epsilon; n.attention_scale=attention_scale; n.keepdim=keepdim;
 return true;
}

bool Graph::setNodeShapeAttributes(uint32_t id, std::vector<uint64_t> reshapeShape,
                                     std::vector<uint64_t> broadcastShape,
                                     std::vector<uint64_t> sliceStarts,
                                     std::vector<uint64_t> sliceLengths, std::string& e) {
 if(id>=nodes_.size()){e="graph node id out of range";return false;}
 if(!reshapeShape.empty() && !TensorShape(reshapeShape).valid()){e="invalid reshape shape";return false;}
 if(!broadcastShape.empty() && !TensorShape(broadcastShape).valid()){e="invalid broadcast shape";return false;}
 if(sliceStarts.size()!=sliceLengths.size()){e="slice starts/lengths size mismatch";return false;}
 nodes_[id].reshape_shape=std::move(reshapeShape); nodes_[id].broadcast_shape=std::move(broadcastShape);
 nodes_[id].slice_starts=std::move(sliceStarts); nodes_[id].slice_lengths=std::move(sliceLengths);
 return true;
}

bool Graph::inferNodeOutput(uint32_t nodeId, tensor::TensorShape& shape, tensor::TensorDType& dtype, std::string& e) const {
 if(nodeId>=nodes_.size()){e="graph node id out of range";return false;}
 const auto& n=nodes_[nodeId]; if(n.outputs.size()!=1){e="shape inference requires exactly one output";return false;}
 if(n.op==OpType::Input){const auto& v=values_[n.outputs[0]];shape=v.shape;dtype=v.dtype;return true;}
 if(n.inputs.empty()){e="shape inference requires an input";return false;}
 const auto& a=values_[n.inputs[0]]; dtype=a.dtype;
 auto get=[&](size_t i)->const GraphValue*{if(i>=n.inputs.size()||n.inputs[i]>=values_.size())return nullptr;return &values_[n.inputs[i]];};
 auto broadcast=[&](const TensorShape&x,const TensorShape&y,TensorShape&out)->bool{size_t r=std::max(x.rank(),y.rank());std::vector<uint64_t>d(r,1);for(size_t i=0;i<r;++i){uint64_t xd=i<x.rank()?x.dim(x.rank()-1-i):1,yd=i<y.rank()?y.dim(y.rank()-1-i):1;if(xd!=yd&&xd!=1&&yd!=1){e="shape inference broadcast mismatch";return false;}d[r-1-i]=std::max(xd,yd);}out=TensorShape(std::move(d));return out.valid();};
 switch(n.op){
  case OpType::Add:case OpType::Sub:case OpType::Mul:case OpType::Div:{const auto*b=get(1);if(!b||a.dtype!=b->dtype){e="shape inference binary dtype mismatch";return false;}return broadcast(a.shape,b->shape,shape);}
  case OpType::Exp:case OpType::Sqrt:case OpType::Rsqrt:case OpType::Clamp:case OpType::SiLU:case OpType::GELU:case OpType::Softmax:case OpType::LayerNorm:case OpType::RMSNorm:case OpType::GroupNorm:case OpType::RoPE: shape=a.shape; break;
  case OpType::Broadcast:{shape=n.broadcast_shape.empty()?values_[n.outputs[0]].shape:TensorShape(n.broadcast_shape);if(!shape.valid())return false;break;}
  case OpType::Slice:{if(n.slice_starts.size()!=a.shape.rank()||n.slice_lengths.size()!=a.shape.rank()){e="Slice requires per-dimension starts and lengths";return false;}auto d=a.shape.dims();for(size_t i=0;i<d.size();++i){if(n.slice_starts[i]>d[i]||n.slice_lengths[i]>d[i]-n.slice_starts[i]){e="Slice range out of bounds";return false;}d[i]=n.slice_lengths[i];}shape=TensorShape(std::move(d));break;}
  case OpType::Reshape:{shape=n.reshape_shape.empty()?values_[n.outputs[0]].shape:TensorShape(n.reshape_shape);if(!shape.valid()||shape.elementCount()!=a.shape.elementCount()){e="Reshape shape is incompatible with input";return false;}break;}
  case OpType::Transpose:{if(n.permutation.size()!=a.shape.rank()){e="Transpose permutation rank mismatch";return false;}std::vector<uint64_t>d(a.shape.rank());std::vector<uint8_t>seen(a.shape.rank());for(size_t i=0;i<n.permutation.size();++i){auto q=n.permutation[i];if(q>=a.shape.rank()||seen[q]){e="invalid Transpose permutation";return false;}seen[q]=1;d[i]=a.shape.dim(q);}shape=TensorShape(std::move(d));break;}
  case OpType::Concat:{if(n.inputs.size()<2||n.axes.empty()||n.axes[0]>=a.shape.rank()){e="Concat axis is missing or invalid";return false;}size_t ax=n.axes[0];auto d=a.shape.dims();uint64_t sum=0;for(auto id:n.inputs){const auto&v=values_[id];if(v.shape.rank()!=a.shape.rank()){e="Concat rank mismatch";return false;}for(size_t i=0;i<d.size();++i)if(i!=ax&&v.shape.dim(i)!=d[i]){e="Concat shape mismatch";return false;}if(sum>UINT64_MAX-v.shape.dim(ax)){e="Concat dimension overflow";return false;}sum+=v.shape.dim(ax);}d[ax]=sum;shape=TensorShape(std::move(d));break;}
  case OpType::Upsample:{if(a.shape.rank()!=4||n.scale_factor==0){e="Upsample requires rank-4 and positive scale";return false;}auto d=a.shape.dims();if(d[2]>UINT64_MAX/n.scale_factor||d[3]>UINT64_MAX/n.scale_factor){e="Upsample shape overflow";return false;}d[2]*=n.scale_factor;d[3]*=n.scale_factor;shape=TensorShape(std::move(d));break;}
  case OpType::MatMul:case OpType::BatchedMatMul:{const auto*b=get(1);if(!b||a.shape.rank()!=b->shape.rank()||a.shape.rank()<2||a.shape.dim(a.shape.rank()-1)!=b->shape.dim(b->shape.rank()-2)){e="MatMul shape mismatch";return false;}auto d=a.shape.dims();d.back()=b->shape.dim(b->shape.rank()-1);for(size_t i=0;i+2<d.size();++i){auto x=a.shape.dim(i),y=b->shape.dim(i);if(x!=y&&x!=1&&y!=1){e="MatMul batch broadcast mismatch";return false;}d[i]=std::max(x,y);}shape=TensorShape(std::move(d));break;}
  case OpType::Linear:{const auto*w=get(1);if(!w||w->shape.rank()!=2||a.shape.rank()<2||a.shape.dim(a.shape.rank()-1)!=w->shape.dim(0)){e="Linear shape mismatch";return false;}auto d=a.shape.dims();d.back()=w->shape.dim(1);shape=TensorShape(std::move(d));break;}
  case OpType::Conv2D:{const auto*w=get(1);if(!w||a.shape.rank()!=4||w->shape.rank()!=4||n.groups==0||a.shape.dim(1)%n.groups||w->shape.dim(1)!=a.shape.dim(1)/n.groups){e="Conv2D shape/groups mismatch";return false;}uint64_t kh=w->shape.dim(2),kw=w->shape.dim(3),effh=1+static_cast<uint64_t>(n.dilation)*(kh-1),effw=1+static_cast<uint64_t>(n.dilation)*(kw-1),nh=a.shape.dim(2)+2*static_cast<uint64_t>(n.padding),nw=a.shape.dim(3)+2*static_cast<uint64_t>(n.padding);if(nh<effh||nw<effw){e="Conv2D kernel exceeds padded input";return false;}auto d=a.shape.dims();d[1]=w->shape.dim(0);d[2]=(nh-effh)/n.stride+1;d[3]=(nw-effw)/n.stride+1;shape=TensorShape(std::move(d));break;}
  case OpType::Attention:{const auto*k=get(1),*v=get(2);if(!k||!v||a.shape.rank()!=k->shape.rank()||a.shape.rank()!=v->shape.rank()||(a.shape.rank()!=3&&a.shape.rank()!=4)||a.shape.dim(a.shape.rank()-1)!=k->shape.dim(k->shape.rank()-1)||k->shape.dim(k->shape.rank()-2)!=v->shape.dim(v->shape.rank()-2)){e="Attention shape mismatch";return false;}auto d=a.shape.dims();d.back()=v->shape.dim(v->shape.rank()-1);shape=TensorShape(std::move(d));break;}
  default:e="shape inference unsupported for "+std::string(opName(n.op));return false;
 }
 return shape.valid();
}

bool Graph::validate(std::string& e) const {
 std::set<uint32_t> producers;
 std::set<std::string> valueNames;
 for(const auto& v:values_) {
  if(v.id >= values_.size() || v.name.empty()){e="invalid graph value metadata";return false;}
  if(!v.shape.valid() || tensor::dtypeSize(v.dtype)==0){e="invalid graph value shape/dtype: "+v.name;return false;}
  if(!valueNames.insert(v.name).second){e="duplicate graph value name: "+v.name;return false;}
 }
 for(const auto& n:nodes_) {
  if(n.id>=nodes_.size()){e="invalid graph node id";return false;}
  if(n.outputs.empty() && n.op!=OpType::Input){e="node has no outputs: "+n.name;return false;}
  for(auto v:n.inputs) if(v>=values_.size()){e="node input value out of range: "+n.name;return false;}
  for(auto v:n.outputs) {
   if(v>=values_.size()){e="node output value out of range: "+n.name;return false;}
   if(!producers.insert(v).second){e="multiple nodes produce value "+std::to_string(v);return false;}
  }
  if(n.op==OpType::Input && !n.outputs.empty() && !n.inputs.empty()){e="Input node cannot have inputs";return false;}
  if(n.op!=OpType::Input){
   size_t minInputs=1,maxInputs=1;
   switch(n.op){
    case OpType::Add: case OpType::Sub: case OpType::Mul: case OpType::Div: case OpType::MatMul: case OpType::BatchedMatMul: minInputs=maxInputs=2; break;
    case OpType::Linear: case OpType::LayerNorm: case OpType::GroupNorm: case OpType::Conv2D: minInputs=2; maxInputs=3; break;
    case OpType::Attention: minInputs=3; maxInputs=4; break;
    case OpType::RoPE: minInputs=maxInputs=2; break;
    case OpType::Concat: minInputs=2; maxInputs=std::numeric_limits<size_t>::max(); break;
    case OpType::Input: break;
    default: break;
   }
   if(n.inputs.size()<minInputs || n.inputs.size()>maxInputs){e="invalid input count for "+std::string(opName(n.op))+": "+n.name;return false;}
  }
 }
 for(const auto& n:nodes_) {
  if(n.op==OpType::Input) {
    if(n.inputs.size()!=0 || n.outputs.size()!=1){e="Input node must have zero inputs and exactly one output";return false;}
  } else {
    for(auto v:n.inputs){
      bool produced=false;
      for(const auto& p:nodes_) for(auto ov:p.outputs) if(ov==v) {produced=true;break;}
      if(!produced){e="graph input value has no producer: "+std::to_string(v);return false;}
    }
  }
}
 std::set<std::string> names;
 for(const auto& v:values_) if(v.name.empty() || !names.insert(v.name).second){e="duplicate or empty graph value name";return false;}
 for(const auto& n:nodes_) if(n.op!=OpType::Input){
  if(n.outputs.size()!=1){e="shape inference requires exactly one output: "+n.name;return false;}
  tensor::TensorShape inferred; tensor::TensorDType inferredDtype=tensor::TensorDType::Unknown;
  if(!inferNodeOutput(n.id,inferred,inferredDtype,e)) return false;
  const auto& declared=values_[n.outputs[0]];
  if(declared.shape.dims()!=inferred.dims()){e="graph output shape does not match inferred shape: "+n.name;return false;}
  if(declared.dtype!=inferredDtype){e="graph output dtype does not match inferred dtype: "+n.name;return false;}
 }
 std::vector<uint32_t> order;
 return topologicalSort(order,e);
}

bool Graph::topologicalSort(std::vector<uint32_t>& order,std::string& e) const {
 order.clear();
 std::vector<std::vector<uint32_t>> adj(nodes_.size());
 std::vector<uint32_t> indeg(nodes_.size());
 std::unordered_map<uint32_t,uint32_t> producer;
 for(const auto& n:nodes_) {
  for(auto v:n.outputs) {
   if(v>=values_.size()){e="graph output value out of range";return false;}
   if(!producer.emplace(v,n.id).second){e="multiple nodes produce the same graph value";return false;}
  }
 }
 for(const auto& n:nodes_) for(auto v:n.inputs) {
  if(v>=values_.size()){e="graph input value out of range";return false;}
  auto it=producer.find(v);
  if(it!=producer.end() && it->second!=n.id) {adj[it->second].push_back(n.id); ++indeg[n.id];}
 }
 std::queue<uint32_t> q;
 for(uint32_t i=0;i<nodes_.size();++i) if(!indeg[i]) q.push(i);
 while(!q.empty()){
  auto n=q.front();q.pop();order.push_back(n);
  for(auto d:adj[n]) if(--indeg[d]==0) q.push(d);
 }
 if(order.size()!=nodes_.size()){e="graph contains a cycle or unresolved dependency";order.clear();return false;}
 return true;
}

bool TensorRegistry::put(const std::string& n,const Tensor& t,std::string& e) {
 if(n.empty()){e="tensor name is empty";return false;}
 if(!t.valid()){e="cannot register invalid tensor";return false;}
 tensors_[n]=t; return true;
}
bool TensorRegistry::get(const std::string& n,Tensor& o,std::string& e) const {
 auto it=tensors_.find(n); if(it==tensors_.end()){e="tensor not found: "+n;return false;}
 o=it->second; return true;
}
bool TensorRegistry::has(const std::string& n) const {return tensors_.find(n)!=tensors_.end();}

bool ExecutionPlanner::build(const Graph& g,std::vector<ExecutionStep>& p,std::string& e) const {
 std::vector<uint32_t> order; if(!g.validate(e)) return false;
 if(!g.topologicalSort(order,e)) return false;
 p.clear(); p.reserve(order.size());
 for(auto id:order){const auto& n=g.nodes()[id];p.push_back({id,n.op,n.name});}
 return true;
}

static bool materializeCpuInputs(const std::vector<Tensor>& in, std::vector<Tensor>& f32in, std::string& e) {
 if(in.empty()){e="CPU operator requires at least one input";return false;}
 f32in.clear(); f32in.reserve(in.size());
 TensorRuntime rt;
 for(const auto& t:in){
  if(!t.valid()){e="CPU operator received an invalid tensor";return false;}
  Tensor x;
  if(!rt.convertDtype(t,TensorDType::F32,x,e)) return false;
  if(!x.isContiguous()){e="CPU materialization produced a non-contiguous tensor";return false;}
  f32in.push_back(std::move(x));
 }
 return true;
}
static bool restoreCpuDtype(const std::vector<Tensor>& in, Tensor& out, std::string& e) {
 if(in.empty()) { e="cannot infer CPU output dtype without inputs"; return false; }
 const TensorDType target=in.front().dtype();
 if(target==TensorDType::F32){return true;}
 Tensor converted;
 if(!TensorRuntime().convertDtype(out,target,converted,e)) return false;
 out=std::move(converted);
 return true;
}

static bool require(const std::vector<Tensor>& in,size_t n,const char* op,std::string& e){
 if(in.size()!=n){e=std::string(op)+" requires "+std::to_string(n)+" inputs";return false;}
 for(const auto& t:in) if(!t.valid()){e=std::string(op)+" received an invalid tensor";return false;}
 return true;
}

bool CPUBackend::execute(OpType op,const std::vector<Tensor>& in,Tensor& o,std::string& e, const GraphNode* node) const {
 std::vector<Tensor> f32in;
 if(!materializeCpuInputs(in,f32in,e)) return false;
 using namespace ops;
 bool ok=false;
 switch(op){
  case OpType::Add: if(!require(f32in,2,"Add",e))return false; ok=add(f32in[0],f32in[1],o,e); break;
  case OpType::Sub: if(!require(f32in,2,"Sub",e))return false; ok=sub(f32in[0],f32in[1],o,e); break;
  case OpType::Mul: if(!require(f32in,2,"Mul",e))return false; ok=mul(f32in[0],f32in[1],o,e); break;
  case OpType::Div: if(!require(f32in,2,"Div",e))return false; ok=div(f32in[0],f32in[1],o,e); break;
  case OpType::MatMul: if(!require(f32in,2,"MatMul",e))return false; ok=matmul(f32in[0],f32in[1],o,e); break;
  case OpType::Linear: if(f32in.size()!=2&&f32in.size()!=3){e="Linear requires input, weight and optional bias";return false;} ok=linear(f32in[0],f32in[1],f32in.size()==3?&f32in[2]:nullptr,o,e); break;
  case OpType::Exp: if(!require(f32in,1,"Exp",e))return false; ok=unary(f32in[0],o,"exp",e); break;
  case OpType::Sqrt: if(!require(f32in,1,"Sqrt",e))return false; ok=unary(f32in[0],o,"sqrt",e); break;
  case OpType::Rsqrt: if(!require(f32in,1,"Rsqrt",e))return false; ok=unary(f32in[0],o,"rsqrt",e); break;
  case OpType::Clamp: if(!require(f32in,1,"Clamp",e))return false; ok=unary(f32in[0],o,"clamp",e); break;
  case OpType::BatchedMatMul: if(!require(f32in,2,"BatchedMatMul",e))return false; ok=batchedMatmul(f32in[0],f32in[1],o,e); break;
  case OpType::RMSNorm: if(!require(f32in,2,"RMSNorm",e))return false; ok=rmsNorm(f32in[0],f32in[1],node?node->epsilon:1e-5,o,e); break;
  case OpType::LayerNorm: if(f32in.size()!=2&&f32in.size()!=3){e="LayerNorm requires input, gamma and optional beta";return false;} ok=layerNorm(f32in[0],f32in[1],f32in.size()==3?&f32in[2]:nullptr,node?node->epsilon:1e-5,o,e); break;
  case OpType::GroupNorm: if(f32in.size()!=2&&f32in.size()!=3){e="GroupNorm requires input, gamma and optional beta";return false;} ok=groupNorm(f32in[0],f32in[1],f32in.size()==3?&f32in[2]:nullptr,node?node->groups:32,node?node->epsilon:1e-5,o,e); break;
  case OpType::Conv2D: if(f32in.size()!=2&&f32in.size()!=3){e="Conv2D requires input, weight and optional bias";return false;} ok=conv2d(f32in[0],f32in[1],f32in.size()==3?&f32in[2]:nullptr,node?node->stride:1,node?node->padding:0,node?node->dilation:1,node?node->groups:1,o,e); break;
  case OpType::Softmax: if(!require(f32in,1,"Softmax",e))return false; ok=softmax(f32in[0],node&&!node->axes.empty()?node->axes[0]:f32in[0].shape().rank()-1,o,e); break;
  case OpType::Transpose: if(!require(f32in,1,"Transpose",e))return false; {std::vector<size_t> p=node?node->permutation:std::vector<size_t>{}; if(p.empty()){p.resize(f32in[0].shape().rank());for(size_t i=0;i<p.size();++i)p[i]=p.size()-1-i;} ok=transpose(f32in[0],p,o,e);} break;
  case OpType::Upsample: if(!require(f32in,1,"Upsample",e))return false; ok=upsampleNearest(f32in[0],node?node->scale_factor:2,o,e); break;
  case OpType::Concat: if(f32in.empty()){e="Concat requires inputs";return false;} ok=concat(f32in,node&&!node->axes.empty()?node->axes[0]:0,o,e); break;
  case OpType::Attention: { if(f32in.size()!=3&&f32in.size()!=4){e="Attention requires Q, K, V and optional mask";return false;} const size_t r=f32in[0].shape().rank(); if(r!=3&&r!=4){e="Attention Q must be rank 3 or rank 4";return false;} const double scale=node&&node->attention_scale>0?node->attention_scale:1.0/std::sqrt((double)f32in[0].shape().dim(r-1)); ok=scaledDotProductAttention(f32in[0],f32in[1],f32in[2],scale,f32in.size()==4?&f32in[3]:nullptr,o,e); break; }
  case OpType::SiLU: if(!require(f32in,1,"SiLU",e))return false; ok=unary(f32in[0],o,"silu",e); break;
  case OpType::GELU: if(!require(f32in,1,"GELU",e))return false; ok=unary(f32in[0],o,"gelu",e); break;
  case OpType::Slice: if(!require(f32in,1,"Slice",e))return false; if(!node||node->slice_starts.size()!=f32in[0].shape().rank()||node->slice_lengths.size()!=f32in[0].shape().rank()){e="Slice requires per-dimension starts and lengths";return false;} {Tensor current=f32in[0]; for(size_t d=0;d<node->slice_starts.size();++d){Tensor next; if(!TensorRuntime().slice(current,d,node->slice_starts[d],node->slice_lengths[d],next,e))return false; current=std::move(next);} o=std::move(current); ok=true;} break;
  case OpType::Broadcast: if(!require(f32in,1,"Broadcast",e))return false; if(!node||node->broadcast_shape.empty()){e="Broadcast requires broadcast_shape";return false;} {TensorShape target(node->broadcast_shape); if(!target.valid()){e=target.error();return false;} ok=broadcastTo(f32in[0],target,o,e);} break;
  case OpType::Reshape: e="Reshape must be handled by Graph::execute with graph output metadata"; return false;
  case OpType::RoPE: if(!require(f32in,2,"RoPE",e))return false; { Tensor tmp; ok=ops::rope(f32in[0],f32in[1],tmp,e); o=std::move(tmp); } break;
  case OpType::Input: e="Input is a graph source and is not a backend operation"; return false;
 }
 if(!ok)return false;
 return restoreCpuDtype(in,o,e);
}

VulkanBackend::VulkanBackend() {
#ifndef LOCALIMAGE_NO_VULKAN
 context_ = std::make_unique<localimage::VulkanContext>();
 std::string e;
 if (context_->initialize(e)) compute_ = std::make_unique<localimage::vulkan::VulkanCompute>(*context_);
#endif
}
VulkanBackend::~VulkanBackend() = default;
bool VulkanBackend::available(std::string& e) const {
#ifndef LOCALIMAGE_NO_VULKAN
 if (!context_ || !compute_ || !context_->device()) { e = "Vulkan backend unavailable"; return false; }
 return compute_->supported(e);
#else
 e = "Vulkan backend not compiled in host reference build"; return false;
#endif
}
bool VulkanBackend::execute(OpType op,const std::vector<Tensor>& inputs,Tensor& output,const GraphNode& node,std::string& e) const {
#ifndef LOCALIMAGE_NO_VULKAN
 if (!compute_) { e = "Vulkan backend unavailable"; return false; }
 ir::Op iop = ir::Op::Input;
 switch(op){
  case OpType::Add:iop=ir::Op::Add;break; case OpType::Sub:iop=ir::Op::Sub;break; case OpType::Mul:iop=ir::Op::Mul;break; case OpType::Div:iop=ir::Op::Div;break;
  case OpType::Exp:iop=ir::Op::Exp;break; case OpType::Sqrt:iop=ir::Op::Sqrt;break; case OpType::Rsqrt:iop=ir::Op::Rsqrt;break;
  case OpType::GELU:iop=ir::Op::GELU;break; case OpType::SiLU:iop=ir::Op::SiLU;break; case OpType::Clamp:iop=ir::Op::Clamp;break;
  case OpType::MatMul:iop=ir::Op::MatMul;break; case OpType::BatchedMatMul:iop=ir::Op::BatchedMatMul;break;
  case OpType::Linear:iop=ir::Op::Linear;break; case OpType::Conv2D:iop=ir::Op::Conv2D;break; case OpType::Softmax:iop=ir::Op::Softmax;break;
  case OpType::LayerNorm:iop=ir::Op::LayerNorm;break; case OpType::RMSNorm:iop=ir::Op::RMSNorm;break; case OpType::GroupNorm:iop=ir::Op::GroupNorm;break; case OpType::Attention:iop=ir::Op::Attention;break;
  case OpType::Transpose:iop=ir::Op::Transpose;break; case OpType::Slice:iop=ir::Op::Slice;break; case OpType::Broadcast:iop=ir::Op::Broadcast;break; case OpType::Concat:iop=ir::Op::Concat;break; case OpType::Upsample:iop=ir::Op::Upsample;break;
  default:e="Vulkan operator unavailable: "+std::string(opName(op));return false;
 }
 ir::Attributes a; a.axes=node.axes; a.permutation=node.permutation; a.groups=node.groups; a.stride=node.stride; a.padding=node.padding; a.dilation=node.dilation; a.scale_factor=node.scale_factor; a.epsilon=node.epsilon; a.attention_scale=node.attention_scale; a.keepdim=node.keepdim; a.reshape_shape=node.reshape_shape; a.broadcast_shape=node.broadcast_shape; a.slice_starts=node.slice_starts; a.slice_lengths=node.slice_lengths;
 return compute_->execute(iop,inputs,a,output,e);
#else
 (void)op;(void)inputs;(void)output;(void)node;e="Vulkan backend not compiled in host reference build";return false;
#endif
}

bool Graph::execute(const TensorRegistry& inputs, TensorRegistry& outputs, bool preferVulkan,
                   std::string& backend, std::string& e) const {
 if(!validate(e)) return false;
 std::vector<uint32_t> order; if(!topologicalSort(order,e)) return false;
 CPUBackend cpu;
 VulkanBackend vk;
 std::string vkError;
 const bool vkReady = preferVulkan && vk.available(vkError);
 bool usedVulkan=false, usedCpu=false;
 backend = vkReady ? "Vulkan" : "CPU";
 TensorRegistry values;
 for(const auto& n:nodes_) if(n.op==OpType::Input) {
  if(n.outputs.size()!=1){e="Input node must have exactly one output";return false;}
  const auto& v=values_[n.outputs[0]];
  Tensor t; if(!inputs.get(v.name,t,e)) return false;
  if(t.dtype()!=v.dtype || t.shape().dims()!=v.shape.dims()){e="input tensor shape/dtype mismatch: "+v.name;return false;}
  if(!values.put(v.name,t,e))return false;
 }
 for(auto id:order){
  const auto& n=nodes_[id]; if(n.op==OpType::Input) continue;
  if(n.outputs.size()!=1){e="only single-output operators are supported by this execution API: "+n.name;return false;}
  std::vector<Tensor> ins; ins.reserve(n.inputs.size());
  for(auto vid:n.inputs){Tensor t;if(!values.get(values_[vid].name,t,e))return false;ins.push_back(t);}
  Tensor out;
  if(n.op==OpType::Reshape){
   if(ins.size()!=1){e="Reshape requires exactly one input";return false;}
   if(n.outputs.size()!=1){e="Reshape requires exactly one output";return false;}
   if(!TensorRuntime().reshape(ins[0],values_[n.outputs[0]].shape,out,e)) return false;
  } else {
   std::string gpuError;
   bool done = vkReady && vk.execute(n.op,ins,out,n,gpuError);
   if(done) { usedVulkan = true; }
   else if(vkReady) {
     // Only a positively identified UNSUPPORTED operator may fall back. A
     // pipeline/device/execution failure must be surfaced to the caller.
     const bool unsupported = gpuError.rfind("Vulkan operator unavailable:", 0) == 0;
     if(!unsupported) { e = gpuError.empty() ? "Vulkan execution failed" : gpuError; return false; }
     if(!cpu.execute(n.op,ins,out,e,&n)) return false;
     usedCpu = true;
   } else {
     if(!cpu.execute(n.op,ins,out,e,&n)) return false;
     usedCpu = true;
   }
  }
  const auto& v=values_[n.outputs[0]];
  if(out.shape().dims()!=v.shape.dims() || out.dtype()!=v.dtype){e="operator output metadata mismatch: "+n.name;return false;}
  if(!values.put(v.name,out,e))return false;
 }
 for(const auto& v:values_) if(values.has(v.name)){Tensor t;if(values.get(v.name,t,e))outputs.put(v.name,t,e);}
 if(usedVulkan && usedCpu) backend="Hybrid";
 else if(usedVulkan) backend="Vulkan";
 else backend="CPU";
 return true;
}

} // namespace localimage::graph
