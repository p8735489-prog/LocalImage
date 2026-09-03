#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace localimage::tokenizer {
class BPETokenizer final {
public:
 bool load(const std::string& vocab_json,const std::string& merges_txt,std::string& error);
 bool encode(const std::string& text,size_t max_length,std::vector<uint32_t>& ids,std::string& error) const;
 bool loaded() const{return !vocab_.empty()&&!merges_.empty();}
private:
 std::unordered_map<std::string,uint32_t> vocab_;
 std::unordered_map<std::string,uint32_t> merges_;
 uint32_t unk_=0,bos_=0,eos_=0,pad_=0;
};
}
