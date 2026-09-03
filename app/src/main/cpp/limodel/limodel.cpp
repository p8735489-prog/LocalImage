#include "limodel.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <functional>
namespace localimage::limodel {
std::string manifestJson(const Manifest&m){std::ostringstream s;s<<"{\"format\":\"limodel\",\"version\":"<<m.version<<",\"architecture\":\""<<m.architecture<<"\",\"dtype\":\""<<m.dtype<<"\",\"quantization\":\""<<m.quantization<<"\",\"model_hash\":\""<<m.model_hash<<"\",\"components\":{"<<"\"tokenizer\":"<<(m.tokenizer?"true":"false")<<",\"clip_l\":"<<(m.clip_l?"true":"false")<<",\"openclip_g\":"<<(m.openclip_g?"true":"false")<<",\"unet\":"<<(m.unet?"true":"false")<<",\"vae\":"<<(m.vae?"true":"false")<<"},\"backends\":[";for(size_t i=0;i<m.backends.size();++i){if(i)s<<',';s<<'\"'<<m.backends[i]<<'\"';}s<<"]}";return s.str();}
static bool writeAll(int fd,const void*p,size_t n,std::string&e){const uint8_t*b=(const uint8_t*)p;while(n){ssize_t w=::write(fd,b,n);if(w<=0){e="limodel write failed: "+std::string(std::strerror(errno));return false;}b+=w;n-=w;}return true;}
bool Writer::convert(const safetensors::SafeTensorFile& f,const std::string& path,const Manifest& m,std::string& e)const{
 if(f.tensors().empty()){e="cannot convert empty SafeTensors";return false;}
 if(path.empty()){e="limodel path is empty";return false;}
 if(::mkdir(path.c_str(),0700)!=0&&errno!=EEXIST){e="cannot create limodel directory: "+std::string(std::strerror(errno));return false;}
 const std::string graphDir=path+"/graph",metaDir=path+"/metadata",weightsDir=path+"/weights";
 if((::mkdir(graphDir.c_str(),0700)!=0&&errno!=EEXIST)||(::mkdir(metaDir.c_str(),0700)!=0&&errno!=EEXIST)||(::mkdir(weightsDir.c_str(),0700)!=0&&errno!=EEXIST)){e="cannot create limodel subdirectories: "+std::string(std::strerror(errno));return false;}
 auto writeText=[&](const std::string& file,const std::string& text)->bool{int fd=::open(file.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0600);if(fd<0){e="cannot create "+file+": "+std::string(std::strerror(errno));return false;}bool ok=writeAll(fd,text.data(),text.size(),e);::close(fd);return ok;};
 if(!writeText(path+"/manifest.json",manifestJson(m)))return false;
 std::ostringstream idx; idx<<"# name\tdtype\tshape\tbytes\toffset\n";
 for(const auto&kv:f.tensors()){
     const auto&i=kv.second;
     // Store each tensor as an independent shard. This keeps the package self-contained
     // while allowing future readers to mmap only the tensors they actually materialize.
     std::string shardName=std::to_string(std::hash<std::string>{}(i.name))+".bin";
     std::string shardPath=weightsDir+"/"+shardName;
     safetensors::TensorView view;
     if(!f.getTensorView(i.name,view,e)) return false;
     if(view.byteSize()!=i.byte_size || !view.data()){e="invalid tensor view while converting: "+i.name;return false;}
     int wfd=::open(shardPath.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0600);
     if(wfd<0){e="cannot create weight shard: "+std::string(std::strerror(errno));return false;}
     bool wok=writeAll(wfd,view.data(),static_cast<size_t>(view.byteSize()),e);
     ::close(wfd);
     if(!wok)return false;
     idx<<i.name<<"\t"<<safetensors::dtypeName(i.dtype)<<"\t[";
     for(size_t d=0;d<i.shape.size();++d){if(d)idx<<",";idx<<i.shape[d];}
     idx<<"]\t"<<i.byte_size<<"\t"<<shardName<<"\n";
 }
 if(!writeText(graphDir+"/weights.index",idx.str()))return false;
 if(!writeText(metaDir+"/source.json",std::string("{\"tensor_count\":")+std::to_string(f.tensors().size())+",\"file_size\":"+std::to_string(f.fileSize())+",\"header_size\":"+std::to_string(f.headerSize())+"}"))return false;
 return true;
}

bool Reader::readManifest(const std::string&path,Manifest&o,std::string&e)const{std::string p=path+"/manifest.json";int fd=::open(p.c_str(),O_RDONLY);if(fd<0){e="limodel manifest not found";return false;}char b[8192];ssize_t n=::read(fd,b,sizeof(b)-1);::close(fd);if(n<=0){e="limodel manifest is empty";return false;}b[n]=0;std::string s(b,(size_t)n);if(s.find("\"format\":\"limodel\"")==std::string::npos){e="unsupported limodel manifest";return false;}auto arch=s.find("\"architecture\":\"");if(arch!=std::string::npos){arch+=16;auto end=s.find('"',arch);if(end!=std::string::npos)o.architecture=s.substr(arch,end-arch);}auto ver=s.find("\"version\":");if(ver==std::string::npos){e="limodel version missing";return false;}ver+=10;uint64_t parsed=0;size_t digits=0;while(ver<s.size()&&s[ver]>='0'&&s[ver]<='9'){uint32_t d=static_cast<uint32_t>(s[ver]-'0');if(parsed>(UINT32_MAX-d)/10){e="limodel version overflow";return false;}parsed=parsed*10+d;++ver;++digits;}if(digits==0){e="invalid limodel version";return false;}o.version=static_cast<uint32_t>(parsed);if(o.version!=1){e="unsupported limodel version: "+std::to_string(o.version);return false;}auto mh=s.find("\"model_hash\":\"");if(mh!=std::string::npos){mh+=14;auto me=s.find('"',mh);if(me!=std::string::npos)o.model_hash=s.substr(mh,me-mh);}return true;}
bool Reader::valid(const std::string&path,std::string&e)const{Manifest m;return readManifest(path,m,e);}
}
