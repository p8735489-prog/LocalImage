#include "weight_store.h"
#include <limits>

namespace localimage::runtime {

bool WeightStore::attach(const safetensors::SafeTensorFile& file, std::string& error) {
    if (file.tensors().empty()) { error="cannot attach empty SafeTensors"; return false; }
    entries_.clear(); lru_.clear(); resident_.clear(); total_bytes_=0; file_=&file;
    for (const auto& kv:file.tensors()) {
        if (kv.first.empty() || kv.second.byte_size > std::numeric_limits<uint64_t>::max()-total_bytes_) {
            clear(); error="weight metadata size overflow"; return false;
        }
        entries_.emplace(kv.first,Entry{&kv.second,0}); total_bytes_+=kv.second.byte_size;
    }
    return true;
}
bool WeightStore::contains(const std::string& name) const { return entries_.find(name)!=entries_.end(); }
bool WeightStore::view(const std::string& name,safetensors::TensorView& out,std::string& error) const {
    if(!file_){error="weight store is not attached";return false;}
    if(!contains(name)){error="weight not found: "+name;return false;}
    return file_->getTensorView(name,out,error);
}
void WeightStore::touch(const std::string& name){auto it=resident_.find(name);if(it!=resident_.end()){lru_.erase(it->second);lru_.push_front(name);it->second=lru_.begin();return;}lru_.push_front(name);resident_[name]=lru_.begin();evict();}
void WeightStore::evict(){while(resident_.size()>resident_limit_&&!lru_.empty()){const std::string name=lru_.back();auto e=entries_.find(name);if(e!=entries_.end()&&e->second.refs!=0){lru_.pop_back();lru_.push_front(name);resident_[name]=lru_.begin();bool any=false;for(const auto&x:entries_)if(x.second.refs==0&&resident_.count(x.first)){any=true;break;}if(!any)break;continue;}resident_.erase(name);lru_.pop_back();}}
bool WeightStore::acquire(const std::string& name,safetensors::TensorView& out,std::string& error){if(!view(name,out,error))return false;auto it=entries_.find(name);++it->second.refs;touch(name);return true;}
bool WeightStore::release(const std::string& name,std::string& error){auto it=entries_.find(name);if(it==entries_.end()){error="weight not found: "+name;return false;}if(it->second.refs==0){error="weight release underflow: "+name;return false;}--it->second.refs;return true;}
void WeightStore::clear(){file_=nullptr;entries_.clear();lru_.clear();resident_.clear();total_bytes_=0;}

} // namespace localimage::runtime
