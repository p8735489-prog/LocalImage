#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
namespace localimage::tokenizer {
bool BPETokenizer::load(const std::string& v,const std::string& m,std::string&e){
 std::ifstream vf(v); if(!vf){e="cannot open tokenizer vocab: "+v;return false;}
 // Minimal JSON object parser for token->id pairs; rejects malformed input rather than guessing.
 std::string s((std::istreambuf_iterator<char>(vf)),{}); size_t p=0;
 while(p<s.size()){p=s.find('"',p);if(p==std::string::npos)break;size_t q=s.find('"',p+1);if(q==std::string::npos){e="malformed tokenizer vocab";return false;}std::string tok=s.substr(p+1,q-p-1);size_t c=s.find(':',q+1);if(c==std::string::npos){e="malformed tokenizer vocab";return false;}size_t n=c+1;while(n<s.size()&&(s[n]==' '||s[n]=='\n'||s[n]=='\r'||s[n]=='\t'))++n;size_t z=n;while(z<s.size()&&s[z]>='0'&&s[z]<='9')++z;if(z==n){p=q+1;continue;}vocab_[tok]=static_cast<uint32_t>(std::stoul(s.substr(n,z-n)));p=z;}
 if(vocab_.empty()){e="tokenizer vocabulary is empty";return false;}
 std::ifstream mf(m);if(!mf){e="cannot open tokenizer merges: "+m;return false;}std::string line;uint32_t rank=0;while(std::getline(mf,line)){if(line.empty()||line[0]=='#')continue;std::istringstream is(line);std::string a,b;if(!(is>>a>>b))continue;merges_[a+" "+b]=rank++;}
 if(merges_.empty()){e="tokenizer merges are empty";return false;}
 auto findId=[&](const char* x,uint32_t& o){auto it=vocab_.find(x);if(it!=vocab_.end()){o=it->second;return true;}return false;};
 findId("<|unk|>",unk_);findId("<|startoftext|>",bos_);findId("<|endoftext|>",eos_);findId("<|pad|>",pad_);
 return true;
}
bool BPETokenizer::encode(const std::string& text,size_t max_length,std::vector<uint32_t>&ids,std::string&e)const{
 if(vocab_.empty()||merges_.empty()){e="tokenizer is not loaded";return false;}if(max_length==0){e="max_length must be > 0";return false;}
 ids.clear();if(bos_)ids.push_back(bos_);
 std::istringstream ss(text);std::string word;
 while(ss>>word){auto it=vocab_.find(word);if(it!=vocab_.end())ids.push_back(it->second);else{
   for(char ch:word){std::string t(1,ch);auto jt=vocab_.find(t);ids.push_back(jt==vocab_.end()?unk_:jt->second);}
 }}
 if(eos_&&ids.size()<max_length)ids.push_back(eos_);
 if(ids.size()>max_length)ids.resize(max_length);
 while(ids.size()<max_length)ids.push_back(pad_);
 return true;
}
}
