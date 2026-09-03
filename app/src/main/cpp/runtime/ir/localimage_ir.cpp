#include "localimage_ir.h"
#include <queue>
#include <unordered_map>
#include <set>
namespace localimage::ir {
const char* opName(Op o){switch(o){case Op::Input:return"Input";case Op::Add:return"Add";case Op::Sub:return"Sub";case Op::Mul:return"Mul";case Op::Div:return"Div";case Op::Exp:return"Exp";case Op::Sqrt:return"Sqrt";case Op::Rsqrt:return"Rsqrt";case Op::GELU:return"GELU";case Op::SiLU:return"SiLU";case Op::Clamp:return"Clamp";case Op::MatMul:return"MatMul";case Op::BatchedMatMul:return"BatchedMatMul";case Op::Linear:return"Linear";case Op::Conv2D:return"Conv2D";case Op::Softmax:return"Softmax";case Op::LayerNorm:return"LayerNorm";case Op::RMSNorm:return"RMSNorm";case Op::GroupNorm:return"GroupNorm";case Op::Reshape:return"Reshape";case Op::Transpose:return"Transpose";case Op::Slice:return"Slice";case Op::Concat:return"Concat";case Op::Broadcast:return"Broadcast";case Op::Upsample:return"Upsample";case Op::Attention:return"Attention";case Op::RoPE:return"RoPE";}return"Unknown";}
uint32_t Graph::addValue(const TensorSpec&s){uint32_t id=(uint32_t)values_.size();values_.push_back({id,s});return id;}
uint32_t Graph::addNode(Op op,std::vector<uint32_t> in,std::string name){uint32_t id=(uint32_t)nodes_.size();nodes_.push_back({id,op,std::move(in),{}, {},std::move(name)});return id;}
bool Graph::setOutputs(uint32_t id,std::vector<uint32_t> o,std::string&e){if(id>=nodes_.size()||o.empty()){e="IR node/output invalid";return false;}std::set<uint32_t>s;for(auto v:o)if(v>=values_.size()||!s.insert(v).second){e="IR output value invalid or duplicated";return false;}nodes_[id].outputs=std::move(o);return true;}
bool Graph::setAttributes(uint32_t id,const Attributes&a,std::string&e){if(id>=nodes_.size()){e="IR node id out of range";return false;}if(a.groups==0||a.stride==0||a.dilation==0||a.scale_factor==0||a.epsilon<=0){e="invalid IR attributes";return false;}nodes_[id].attr=a;return true;}
bool Graph::validate(std::string&e)const{
 std::set<uint32_t>prod;
 std::set<std::string>names;
 for(size_t i=0;i<values_.size();++i){const auto&v=values_[i];if(v.id!=i){e="IR value id is not dense";return false;}if(!v.spec.shape.valid()){e="invalid tensor shape for value "+v.spec.name;return false;}if(tensor::dtypeSize(v.spec.dtype)==0){e="unsupported tensor dtype for value "+v.spec.name;return false;}if(v.spec.name.empty()){e="IR value name is empty";return false;}if(!names.insert(v.spec.name).second){e="duplicate IR value name: "+v.spec.name;return false;}}
 for(size_t i=0;i<nodes_.size();++i){const auto&n=nodes_[i];if(n.id!=i){e="IR node id is not dense";return false;}if(n.op!=Op::Input&&n.outputs.empty()){e="IR node has no outputs: "+n.name;return false;}if(n.op==Op::Input&&(n.inputs.size()!=0||n.outputs.size()!=1)){e="IR Input must have zero inputs and one output";return false;}for(auto v:n.inputs)if(v>=values_.size()){e="IR input value out of range";return false;}for(auto v:n.outputs){if(v>=values_.size()||!prod.insert(v).second){e="IR output value duplicated or invalid";return false;}}
  if(n.op!=Op::Input){size_t minI=1,maxI=1;switch(n.op){case Op::Add:case Op::Sub:case Op::Mul:case Op::Div:case Op::MatMul:case Op::BatchedMatMul:minI=maxI=2;break;case Op::Linear:case Op::LayerNorm:case Op::GroupNorm:case Op::Conv2D:minI=2;maxI=3;break;case Op::Attention:minI=3;maxI=4;break;case Op::RoPE:minI=maxI=2;break;case Op::Concat:minI=2;maxI=SIZE_MAX;break;default:break;}if(n.inputs.size()<minI||n.inputs.size()>maxI){e="IR invalid input count for "+std::string(opName(n.op));return false;}}
 }
 // Every consumed value must have exactly one producer. This prevents a graph
 // from appearing topologically valid while resolving an input at runtime.
 for(const auto& n:nodes_) for(auto v:n.inputs) if(!prod.count(v)){e="IR input value has no producer: "+std::to_string(v);return false;}
 std::vector<uint32_t>o;return topological(o,e);
}
bool Graph::topological(std::vector<uint32_t>&o,std::string&e)const{std::unordered_map<uint32_t,uint32_t>p;std::vector<uint32_t>deg(nodes_.size());std::vector<std::vector<uint32_t>>adj(nodes_.size());for(auto&n:nodes_)for(auto v:n.outputs){if(!p.emplace(v,n.id).second){e="IR multiple producers";return false;}}for(auto&n:nodes_){std::set<uint32_t>deps;for(auto v:n.inputs){auto it=p.find(v);if(it!=p.end()&&it->second!=n.id&&deps.insert(it->second).second){adj[it->second].push_back(n.id);++deg[n.id];}}}std::queue<uint32_t>q;for(uint32_t i=0;i<nodes_.size();++i)if(!deg[i])q.push(i);o.clear();while(!q.empty()){auto x=q.front();q.pop();o.push_back(x);for(auto y:adj[x])if(--deg[y]==0)q.push(y);}if(o.size()!=nodes_.size()){e="IR contains cycle";o.clear();return false;}return true;}
}
