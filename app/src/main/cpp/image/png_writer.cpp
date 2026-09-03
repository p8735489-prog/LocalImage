#include "png_writer.h"
#include <fstream>
#include <cstdint>
#include <vector>
#include <cstring>
namespace localimage::image {
namespace {
uint32_t crc32(const uint8_t*d,size_t n){uint32_t c=0xffffffffu;for(size_t i=0;i<n;i++){c^=d[i];for(int k=0;k<8;k++)c=(c>>1)^((c&1)?0xedb88320u:0);}return c^0xffffffffu;}
void be32(std::vector<uint8_t>&o,uint32_t x){o.push_back(x>>24);o.push_back(x>>16);o.push_back(x>>8);o.push_back(x);}
void chunk(std::vector<uint8_t>&o,const char*t,const std::vector<uint8_t>&d){size_t at=o.size();be32(o,(uint32_t)d.size());o.insert(o.end(),t,t+4);o.insert(o.end(),d.begin(),d.end());uint32_t c=crc32(o.data()+at+4,d.size()+4);be32(o,c);}
}
bool writePng(const tensor::Tensor&rgb,const std::string&path,std::string&e){
 if(!rgb.valid()||rgb.dtype()!=tensor::TensorDType::F32||!rgb.isContiguous()||rgb.shape().rank()!=3||rgb.shape().dim(0)!=3){e="PNG writer expects contiguous F32 CHW RGB tensor";return false;}
 size_t c=3,h=rgb.shape().dim(1),w=rgb.shape().dim(2);if(h==0||w==0||w>0x7fffffff||h>0x7fffffff){e="invalid RGB dimensions";return false;}
 const float*p=(const float*)rgb.data();std::vector<uint8_t> raw((w*3+1)*h);for(size_t y=0;y<h;y++){raw[y*(w*3+1)]=0;for(size_t x=0;x<w;x++)for(size_t k=0;k<3;k++){double v=p[(k*h+y)*w+x];v=v<0?0:v>1?1:v;raw[y*(w*3+1)+1+x*3+k]=(uint8_t)(v*255.0+0.5);}}
 // Deflate stream with stored blocks, valid PNG zlib without third-party dependency.
 std::vector<uint8_t> z={0x78,0x01};size_t pos=0;while(pos<raw.size()){size_t n=std::min<size_t>(65535,raw.size()-pos);bool last=pos+n==raw.size();z.push_back(last?1:0);uint16_t nn=(uint16_t)n;z.push_back(nn);z.push_back(nn>>8);uint16_t inv=(uint16_t)~nn;z.push_back(inv);z.push_back(inv>>8);z.insert(z.end(),raw.begin()+pos,raw.begin()+pos+n);pos+=n;}
 auto adler=[&](){uint32_t a=1,b=0;for(uint8_t x:raw){a=(a+x)%65521;b=(b+a)%65521;}return (b<<16)|a;};be32(z,adler());
 std::vector<uint8_t> out={137,80,78,71,13,10,26,10};std::vector<uint8_t> ihdr;be32(ihdr,(uint32_t)w);be32(ihdr,(uint32_t)h);ihdr.push_back(8);ihdr.push_back(2);ihdr.push_back(0);ihdr.push_back(0);ihdr.push_back(0);chunk(out,"IHDR",ihdr);chunk(out,"IDAT",z);chunk(out,"IEND",{});std::ofstream f(path,std::ios::binary);if(!f){e="cannot create PNG: "+path;return false;}f.write((const char*)out.data(),(std::streamsize)out.size());if(!f){e="PNG write failed";return false;}return true;
}
}
