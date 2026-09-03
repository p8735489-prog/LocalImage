#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>

namespace localimage::tensor {
namespace {
bool checkedMul(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b; return true;
}
bool checkedSize(uint64_t elements, size_t element_size, size_t& out) {
    if (elements > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
    const size_t e = static_cast<size_t>(elements);
    if (element_size != 0 && e > std::numeric_limits<size_t>::max() / element_size) return false;
    out = e * element_size; return true;
}
bool broadcastShape(const TensorShape& a, const TensorShape& b, TensorShape& out, std::string& error) {
    const size_t rank = std::max(a.rank(), b.rank());
    std::vector<uint64_t> dims(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const uint64_t ad = i < a.rank() ? a.dim(a.rank() - 1 - i) : 1;
        const uint64_t bd = i < b.rank() ? b.dim(b.rank() - 1 - i) : 1;
        if (ad != bd && ad != 1 && bd != 1) { error = "incompatible broadcast shapes"; return false; }
        dims[rank - 1 - i] = std::max(ad, bd);
    }
    out = TensorShape(std::move(dims));
    if (!out.valid()) { error = out.error(); return false; }
    return true;
}
uint32_t halfBitsToFloatBits(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) return sign;
        uint32_t m = mant; int e = -14;
        while ((m & 0x400u) == 0) { m <<= 1; --e; }
        m &= 0x3ffu;
        return sign | static_cast<uint32_t>(e + 127) << 23 | (m << 13);
    }
    if (exp == 31) return sign | 0x7f800000u | (mant << 13);
    return sign | ((exp + 112u) << 23) | (mant << 13);
}
float halfToFloat(uint16_t h) { uint32_t bits=halfBitsToFloatBits(h); float f; std::memcpy(&f,&bits,sizeof(f)); return f; }
float bf16ToFloat(uint16_t h) { uint32_t bits=static_cast<uint32_t>(h)<<16; float f; std::memcpy(&f,&bits,sizeof(f)); return f; }
uint16_t floatToBF16(float f) { uint32_t bits; std::memcpy(&bits,&f,sizeof(bits)); uint32_t lsb=(bits>>16)&1u; uint32_t rounding=0x7fffu+lsb; bits+=rounding; return static_cast<uint16_t>(bits>>16); }
uint16_t floatToHalf(float f) {
    uint32_t bits; std::memcpy(&bits,&f,sizeof(bits)); const uint32_t sign=(bits>>16)&0x8000u; int exp=((bits>>23)&0xff)-127+15; uint32_t mant=bits&0x7fffffu;
    if(exp<=0){ if(exp<-10)return static_cast<uint16_t>(sign); mant|=0x800000u; uint32_t shift=static_cast<uint32_t>(14-exp); return static_cast<uint16_t>(sign | ((mant + (1u<<(shift-1)))>>shift)); }
    if(exp>=31)return static_cast<uint16_t>(sign|0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp)<<10) | ((mant+0x1000u)>>13));
}

uint64_t offsetForBroadcast(const Tensor& t, const std::vector<uint64_t>& out_index) {
    uint64_t offset = 0;
    const size_t shift = out_index.size() - t.shape().rank();
    for (size_t d = 0; d < t.shape().rank(); ++d) {
        const uint64_t index = t.shape().dim(d) == 1 ? 0 : out_index[shift + d];
        offset += index * t.stride().stride(d);
    }
    return offset;
}
}

const char* dtypeName(TensorDType dtype) {
    switch (dtype) { case TensorDType::F32: return "F32"; case TensorDType::F16: return "F16"; case TensorDType::BF16: return "BF16"; case TensorDType::I8: return "I8"; case TensorDType::U8: return "U8"; default: return "UNKNOWN"; }
}
size_t dtypeSize(TensorDType dtype) { switch (dtype) { case TensorDType::F32: return 4; case TensorDType::F16: case TensorDType::BF16: return 2; case TensorDType::I8: case TensorDType::U8: return 1; default: return 0; } }
TensorDType fromSafeTensorDType(safetensors::DType dtype) {
    switch (dtype) { case safetensors::DType::F32: return TensorDType::F32; case safetensors::DType::F16: return TensorDType::F16; case safetensors::DType::BF16: return TensorDType::BF16; case safetensors::DType::I8: return TensorDType::I8; case safetensors::DType::U8: return TensorDType::U8; default: return TensorDType::Unknown; }
}

TensorShape::TensorShape(std::vector<uint64_t> dims) : dims_(std::move(dims)) {
    element_count_ = 1;
    for (uint64_t d : dims_) {
        if (!checkedMul(element_count_, d, element_count_)) { valid_ = false; error_ = "shape element count overflow"; return; }
    }
}
uint64_t TensorShape::dim(size_t index) const { return index < dims_.size() ? dims_[index] : 0; }
TensorStride TensorStride::contiguous(const TensorShape& shape) {
    std::vector<uint64_t> v(shape.rank(), 1);
    uint64_t stride = 1;
    for (size_t i = shape.rank(); i-- > 0;) { v[i] = stride; uint64_t next; if (!checkedMul(stride, shape.dim(i), next)) { v[i] = 0; break; } stride = next; }
    return TensorStride(std::move(v));
}
uint64_t TensorStride::stride(size_t dimension) const { return dimension < values_.size() ? values_[dimension] : 0; }
bool TensorStride::isContiguous(const TensorShape& shape) const { return values_ == TensorStride::contiguous(shape).values_; }

TensorStorage::~TensorStorage() {
    if (owned_ && data_) std::free(data_);
}
std::shared_ptr<TensorStorage> TensorStorage::allocate(size_t bytes, size_t alignment, std::string& error) {
    auto storage = std::shared_ptr<TensorStorage>(new TensorStorage());
    storage->size_ = bytes; storage->owned_ = true;
    if (bytes == 0) return storage;
    void* p = nullptr;
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) { error = "alignment must be a power of two"; return nullptr; }
    if (posix_memalign(&p, alignment, bytes) != 0 || !p) { error = "aligned allocation failed"; return nullptr; }
    storage->data_ = p;
    return storage;
}
std::shared_ptr<TensorStorage> TensorStorage::mapped(std::shared_ptr<const safetensors::MappedFile> owner) {
    auto storage = std::shared_ptr<TensorStorage>(new TensorStorage()); storage->owned_ = false; storage->mapped_owner_ = std::move(owner); return storage;
}

bool Tensor::create(const TensorShape& shape, TensorDType dtype, TensorDevice device, Tensor& out, std::string& error) {
    if (!shape.valid()) { error = shape.error(); return false; }
    const size_t es = dtypeSize(dtype); if (!es) { error = "unsupported dtype: " + std::string(dtypeName(dtype)); return false; }
    if (device != TensorDevice::CPU) { error = "only CPU allocation is implemented in Milestone 3"; return false; }
    size_t bytes = 0; if (!checkedSize(shape.elementCount(), es, bytes)) { error = "tensor byte size overflow"; return false; }
    auto storage = TensorStorage::allocate(bytes, 64, error); if (!storage) return false;
    out = Tensor{}; out.storage_ = std::move(storage); out.data_ = static_cast<const uint8_t*>(out.storage_->data()); out.shape_ = shape; out.stride_ = TensorStride::contiguous(shape); out.dtype_ = dtype; out.device_ = device; out.byte_size_ = bytes; return true;
}
Tensor Tensor::fromView(const safetensors::TensorView& view, std::string& error) {
    Tensor out; TensorShape shape(view.shape()); if (!shape.valid()) { error = shape.error(); return {}; }
    TensorDType dtype = fromSafeTensorDType(view.dtype()); if (dtype == TensorDType::Unknown) { error = "unsupported dtype"; return {}; }
    size_t expected = 0;
    if (!checkedSize(shape.elementCount(), dtypeSize(dtype), expected)) { error = "view byte size overflow"; return {}; }
    if (expected != view.byteSize()) { error = "view byte size does not match shape × dtype"; return {}; }
    if (expected != 0 && view.data() == nullptr) { error = "tensor view data pointer is null"; return {}; }
    if (!view.owner()) { error = "tensor view has no mapped file owner"; return {}; }
    out.storage_ = TensorStorage::mapped(view.owner()); out.data_ = static_cast<const uint8_t*>(view.data()); out.shape_ = std::move(shape); out.stride_ = TensorStride::contiguous(out.shape_); out.dtype_ = dtype; out.device_ = TensorDevice::MAPPED; out.byte_size_ = view.byteSize(); return out;
}
const void* Tensor::data() const { return data_; }
void* Tensor::mutableData() { return storage_ ? storage_->mutableData() : nullptr; }
float Tensor::readFloat32(uint64_t index, std::string& error) const {
    if (index >= shape_.elementCount()) { error = "tensor index out of range"; return 0.0f; }
    if (dtype_ == TensorDType::F32) return reinterpret_cast<const float*>(data_)[index];
    if (dtype_ == TensorDType::F16 || dtype_ == TensorDType::BF16) { uint16_t h; std::memcpy(&h, data_ + index * 2, 2); return dtype_ == TensorDType::F16 ? halfToFloat(h) : bf16ToFloat(h); }
    error = "readFloat32 supports F32/F16/BF16 only"; return 0.0f;
}
bool Tensor::writeFloat32(uint64_t index, float value, std::string& error) {
    if (!mutableData()) { error = "tensor storage is read-only"; return false; }
    if (index >= shape_.elementCount()) { error = "tensor index out of range"; return false; }
    if (dtype_ == TensorDType::F32) { static_cast<float*>(mutableData())[index] = value; return true; }
    if (dtype_ == TensorDType::F16 || dtype_ == TensorDType::BF16) { const uint16_t h=dtype_ == TensorDType::F16 ? floatToHalf(value) : floatToBF16(value); std::memcpy(static_cast<uint8_t*>(mutableData())+index*2,&h,2); return true; }
    error = "writeFloat32 supports F32/F16/BF16 only"; return false;
}
std::string Tensor::reshape(const TensorShape& shape, Tensor& out) const {
    if (!valid() || shape.elementCount() != shape_.elementCount()) return "reshape requires equal element count";
    if (!isContiguous()) return "reshape requires a contiguous tensor";
    if (dtypeSize(dtype_) == 0) return "reshape unsupported dtype";
    out = *this; out.shape_ = shape; out.stride_ = TensorStride::contiguous(shape); return {};
}
std::string Tensor::slice(size_t dimension, uint64_t start, uint64_t length, Tensor& out) const {
    if (!valid()) return "invalid tensor";
    if (dimension >= shape_.rank()) return "slice dimension out of range";
    if (start > shape_.dim(dimension) || length > shape_.dim(dimension) - start) return "slice range out of bounds";
    const size_t es = dtypeSize(dtype_);
    if (es == 0 || stride_.stride(dimension) > UINT64_MAX / es) return "slice byte offset overflow";
    const uint64_t element_offset = start * stride_.stride(dimension);
    if (element_offset > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / es)) return "slice byte offset exceeds size_t";
    out = *this;
    std::vector<uint64_t> dims = shape_.dims(); dims[dimension] = length;
    out.shape_ = TensorShape(std::move(dims));
    if (!out.shape_.valid()) return out.shape_.error();
    if (length == 0) {
        out.byte_size_ = 0;
        return {};
    }
    if (element_offset > UINT64_MAX / es) return "slice byte offset overflow";
    const uint64_t byte_offset = element_offset * es;
    if (byte_offset > byte_size_) return "slice start exceeds source storage";
    out.data_ = data_ + static_cast<size_t>(byte_offset);

    // byte_size_ is the accessible byte span of the view, not merely the
    // logical element count. This matters when slicing a non-last dimension:
    // the last logical element can be separated by the original stride.
    uint64_t max_offset = 0;
    for (size_t i = 0; i < out.shape_.rank(); ++i) {
        const uint64_t d = out.shape_.dim(i);
        const uint64_t st = out.stride_.stride(i);
        if (d > 0 && d - 1 > 0 && st > UINT64_MAX / (d - 1)) return "slice view span overflow";
        const uint64_t term = (d == 0 ? 0 : (d - 1) * st);
        if (term > UINT64_MAX - max_offset) return "slice view span overflow";
        max_offset += term;
    }
    if (max_offset > UINT64_MAX - 1) return "slice view span overflow";
    const uint64_t spanElements = max_offset + 1;
    if (!checkedSize(spanElements, es, out.byte_size_)) return "slice byte span overflow";
    if (out.byte_size_ > byte_size_ - static_cast<size_t>(byte_offset)) return "slice view exceeds source storage";
    return {};
}
Tensor TensorAllocator::allocate(const TensorShape& shape, TensorDType dtype, std::string& error) const { Tensor out; Tensor::create(shape, dtype, TensorDevice::CPU, out, error); return out; }
Tensor TensorRuntime::createTensor(const TensorShape& shape, TensorDType dtype, std::string& error) const { return TensorAllocator{}.allocate(shape, dtype, error); }
Tensor TensorRuntime::createView(const safetensors::TensorView& view, std::string& error) const { return Tensor::fromView(view, error); }
bool TensorRuntime::reshape(const Tensor& input, const TensorShape& shape, Tensor& out, std::string& error) const { error = input.reshape(shape, out); return error.empty(); }
bool TensorRuntime::slice(const Tensor& input, size_t dimension, uint64_t start, uint64_t length, Tensor& out, std::string& error) const { error = input.slice(dimension,start,length,out); return error.empty(); }
bool TensorRuntime::transpose(const Tensor& input, const std::vector<size_t>& perm, Tensor& out, std::string& error) const {
    if (!input.valid()) { error = "transpose requires a valid tensor"; return false; }
    const size_t rank = input.shape().rank();
    if (rank == 0 || perm.size() != rank) { error = "transpose permutation rank mismatch"; return false; }
    std::vector<uint8_t> seen(rank, 0);
    for (size_t p : perm) { if (p >= rank || seen[p]) { error = "invalid transpose permutation"; return false; } seen[p] = 1; }
    std::vector<uint64_t> dims(rank);
    for (size_t i = 0; i < rank; ++i) dims[i] = input.shape().dim(perm[i]);
    TensorShape shape(std::move(dims));
    if (!shape.valid()) { error = shape.error(); return false; }
    const size_t es = dtypeSize(input.dtype());
    if (!es) { error = "transpose unsupported dtype"; return false; }
    out = createTensor(shape, input.dtype(), error);
    if (!out.valid()) return false;
    const uint8_t* src = static_cast<const uint8_t*>(input.data());
    uint8_t* dst = static_cast<uint8_t*>(out.mutableData());
    if (!src || !dst) { error = "transpose tensor storage is unavailable"; return false; }
    for (uint64_t of = 0; of < shape.elementCount(); ++of) {
        uint64_t rem = of, srcOffset = 0;
        std::vector<uint64_t> oi(rank), si(rank);
        for (size_t i = rank; i-- > 0;) { oi[i] = rem % shape.dim(i); rem /= shape.dim(i); }
        for (size_t i = 0; i < rank; ++i) si[perm[i]] = oi[i];
        for (size_t i = 0; i < rank; ++i) {
            const uint64_t st = input.stride().stride(i);
            if (st != 0 && si[i] > UINT64_MAX / st) { error = "transpose source offset overflow"; return false; }
            srcOffset += si[i] * st;
        }
        if (srcOffset > UINT64_MAX / es || srcOffset * es >= input.byteSize() + es) { error = "transpose source byte offset out of bounds"; return false; }
        std::memcpy(dst + static_cast<size_t>(of) * es, src + static_cast<size_t>(srcOffset) * es, es);
    }
    return true;
}


bool TensorRuntime::convertDtype(const Tensor& input, TensorDType dtype, Tensor& out, std::string& error) const {
    if (!input.valid()) { error = "dtype conversion requires a valid tensor"; return false; }
    if (dtypeSize(dtype) == 0) { error = "unsupported destination dtype"; return false; }
    if (input.dtype() == dtype) { out = input; return true; }
    if (input.dtype() == TensorDType::I8 || input.dtype() == TensorDType::U8 ||
        dtype == TensorDType::I8 || dtype == TensorDType::U8) {
        error = "integer dtype conversion is unsupported for implicit floating-point conversion";
        return false;
    }
    out = createTensor(input.shape(), dtype, error);
    if (!out.valid()) return false;
    for (uint64_t i = 0; i < input.shape().elementCount(); ++i) {
        const float value = input.readFloat32(i, error);
        if (!error.empty() || !out.writeFloat32(i, value, error)) return false;
    }
    return true;
}

namespace ops {
template<class F> bool binary(const Tensor& a, const Tensor& b, Tensor& out, std::string& error, F fn) {
    if (!a.valid() || !b.valid()) { error="invalid tensor"; return false; }
    if (a.dtype()!=TensorDType::F32 || b.dtype()!=TensorDType::F32) { error="CPU reference ops currently support F32 only"; return false; }
    TensorShape shape; if (!broadcastShape(a.shape(), b.shape(), shape, error)) return false;
    TensorRuntime rt; out=rt.createTensor(shape,TensorDType::F32,error); if(!error.empty()) return false;
    float* dst=static_cast<float*>(out.mutableData()); const float* ad=static_cast<const float*>(a.data()); const float* bd=static_cast<const float*>(b.data());
    for(uint64_t linear=0; linear<shape.elementCount(); ++linear){ uint64_t rem=linear; std::vector<uint64_t> idx(shape.rank()); for(size_t d=shape.rank(); d-- >0;){idx[d]=shape.dim(d)?rem%shape.dim(d):0; rem=shape.dim(d)?rem/shape.dim(d):0;} const uint64_t ao=offsetForBroadcast(a,idx), bo=offsetForBroadcast(b,idx); dst[linear]=fn(ad[ao],bd[bo]); }
    return true;
}
bool add(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return binary(a,b,o,e,[](float x,float y){return x+y;});}
bool sub(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return binary(a,b,o,e,[](float x,float y){return x-y;});}
bool mul(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){return binary(a,b,o,e,[](float x,float y){return x*y;});}
bool div(const Tensor&a,const Tensor&b,Tensor&o,std::string&e){
    if(!a.valid()||!b.valid()){e="invalid tensor";return false;}
    if(a.dtype()!=TensorDType::F32||b.dtype()!=TensorDType::F32){e="CPU div currently supports F32 only";return false;}
    TensorShape shape; if(!broadcastShape(a.shape(),b.shape(),shape,e)) return false;
    TensorRuntime rt; o=rt.createTensor(shape,TensorDType::F32,e); if(!o.valid()) return false;
    const float*A=(const float*)a.data(); const float*B=(const float*)b.data(); float*O=(float*)o.mutableData();
    for(uint64_t i=0;i<shape.elementCount();++i){
        uint64_t rem=i; std::vector<uint64_t> idx(shape.rank());
        for(size_t d=shape.rank();d-->0;){idx[d]=rem%shape.dim(d);rem/=shape.dim(d);}
        const uint64_t ao=offsetForBroadcast(a,idx), bo=offsetForBroadcast(b,idx);
        if(B[bo]==0.0f){e="division by zero";o=Tensor{};return false;}
        O[i]=A[ao]/B[bo];
        if(!std::isfinite(O[i])){e="division produced non-finite result";o=Tensor{};return false;}
    }
    return true;
}
}
} // namespace localimage::tensor
