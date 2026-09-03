#include "vulkan_tensor.h"
#include <cstring>
#include <limits>
#include <utility>
namespace localimage::vulkan {
namespace { std::string vr(VkResult r){return std::to_string((int)r);} bool submitCopy(VkDevice d,VkQueue q,uint32_t family,VkBuffer src,VkBuffer dst,VkDeviceSize size,std::string&e){VkCommandPool cp=VK_NULL_HANDLE;VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;VkResult r=vkCreateCommandPool(d,&pi,nullptr,&cp);if(r!=VK_SUCCESS){e="vkCreateCommandPool: "+vr(r);return false;}VkCommandBuffer cb{};VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=cp;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;r=vkAllocateCommandBuffers(d,&ai,&cb);if(r==VK_SUCCESS){VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;r=vkBeginCommandBuffer(cb,&bi);}if(r==VK_SUCCESS){VkBufferCopy c{0,0,size};vkCmdCopyBuffer(cb,src,dst,1,&c);r=vkEndCommandBuffer(cb);}VkFence f=VK_NULL_HANDLE;if(r==VK_SUCCESS){VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};r=vkCreateFence(d,&fi,nullptr,&f);}if(r==VK_SUCCESS){VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cb;r=vkQueueSubmit(q,1,&si,f);if(r==VK_SUCCESS)r=vkWaitForFences(d,1,&f,VK_TRUE,UINT64_MAX);}if(r!=VK_SUCCESS)e="Vulkan staging copy: "+vr(r);if(f)vkDestroyFence(d,f,nullptr);vkDestroyCommandPool(d,cp,nullptr);return r==VK_SUCCESS;}}
VulkanTensor::~VulkanTensor(){destroy();}
VulkanTensor::VulkanTensor(VulkanTensor&& other) noexcept
    : device_(other.device_), physical_(other.physical_), queueFamily_(other.queueFamily_),
      gpuBuffer_(other.gpuBuffer_), gpuMemory_(other.gpuMemory_), shape_(std::move(other.shape_)), dtype_(other.dtype_) {
    other.device_=VK_NULL_HANDLE; other.physical_=VK_NULL_HANDLE; other.queueFamily_=VK_QUEUE_FAMILY_IGNORED;
    other.gpuBuffer_=VK_NULL_HANDLE; other.gpuMemory_=VK_NULL_HANDLE; other.dtype_=tensor::TensorDType::Unknown;
}
VulkanTensor& VulkanTensor::operator=(VulkanTensor&& other) noexcept {
    if(this==&other) return *this;
    destroy();
    device_=other.device_; physical_=other.physical_; queueFamily_=other.queueFamily_;
    gpuBuffer_=other.gpuBuffer_; gpuMemory_=other.gpuMemory_; shape_=std::move(other.shape_); dtype_=other.dtype_;
    other.device_=VK_NULL_HANDLE; other.physical_=VK_NULL_HANDLE; other.queueFamily_=VK_QUEUE_FAMILY_IGNORED;
    other.gpuBuffer_=VK_NULL_HANDLE; other.gpuMemory_=VK_NULL_HANDLE; other.dtype_=tensor::TensorDType::Unknown;
    return *this;
}
VkDeviceSize VulkanTensor::byteSize()const{
    const size_t es=tensor::dtypeSize(dtype_);
    if(!es || shape_.elementCount()>std::numeric_limits<VkDeviceSize>::max()/es) return 0;
    return static_cast<VkDeviceSize>(shape_.elementCount())*es;
}
bool VulkanTensor::allocate(VkPhysicalDevice p,VkDevice d,uint32_t qf,const tensor::TensorShape& shape,tensor::TensorDType dtype,std::string&e){
    destroy();
    if(!shape.valid()||tensor::dtypeSize(dtype)==0){e="VulkanTensor allocation requires valid shape and dtype";return false;}
    if(shape.elementCount()>std::numeric_limits<VkDeviceSize>::max()/tensor::dtypeSize(dtype)){e="VulkanTensor byte size overflow";return false;}
    const VkDeviceSize bytes = byteSize();
    if (bytes == 0) { e = "VulkanTensor allocation size is invalid or zero"; return false; }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(p, &properties);
    if (properties.limits.maxStorageBufferRange > 0 && bytes > properties.limits.maxStorageBufferRange) {
        e = "VulkanTensor exceeds maxStorageBufferRange";
        return false;
    }
    physical_=p; device_=d; queueFamily_=qf; shape_=shape; dtype_=dtype;
    VulkanMemoryAllocator a(p,d,qf);
    if(!a.createDeviceLocal(byteSize(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,gpuBuffer_,gpuMemory_,e)){destroy();return false;}
    return true;
}
bool VulkanTensor::upload(VkPhysicalDevice p,VkDevice d,uint32_t qf,const tensor::Tensor&t,VkQueue q,std::string&e){destroy();if(!t.valid()||!t.isContiguous()||tensor::dtypeSize(t.dtype())==0){e="VulkanTensor upload requires valid contiguous supported tensor";return false;}if(!allocate(p,d,qf,t.shape(),t.dtype(),e)) return false; VulkanMemoryAllocator a(p,d,qf); VkBuffer sb=VK_NULL_HANDLE;VkDeviceMemory sm=VK_NULL_HANDLE;if(!a.createStaging(t.byteSize(),sb,sm,e)){destroy();return false;}void* mapped=nullptr;VkResult r=vkMapMemory(d,sm,0,t.byteSize(),0,&mapped);if(r!=VK_SUCCESS){e="vkMapMemory staging upload: "+vr(r);VulkanMemoryAllocator::destroy(d,sb,sm);destroy();return false;}std::memcpy(mapped,t.data(),t.byteSize());vkUnmapMemory(d,sm);bool ok=submitCopy(d,q,qf,sb,gpuBuffer_,t.byteSize(),e);VulkanMemoryAllocator::destroy(d,sb,sm);if(!ok){destroy();return false;}return true;}
bool VulkanTensor::download(VkQueue q,tensor::Tensor&out,std::string&e)const{if(!resident()){e="VulkanTensor is not resident";return false;}out=tensor::TensorRuntime().createTensor(shape_,dtype_,e);if(!out.valid())return false;VulkanMemoryAllocator a(physical_,device_,queueFamily_);VkBuffer sb=VK_NULL_HANDLE;VkDeviceMemory sm=VK_NULL_HANDLE;if(!a.createStaging(byteSize(),sb,sm,e))return false;if(!submitCopy(device_,q,queueFamily_,gpuBuffer_,sb,byteSize(),e)){VulkanMemoryAllocator::destroy(device_,sb,sm);return false;}void* mapped=nullptr;VkResult r=vkMapMemory(device_,sm,0,byteSize(),0,&mapped);if(r!=VK_SUCCESS){e="vkMapMemory staging download: "+vr(r);VulkanMemoryAllocator::destroy(device_,sb,sm);return false;}std::memcpy(out.mutableData(),mapped,byteSize());vkUnmapMemory(device_,sm);VulkanMemoryAllocator::destroy(device_,sb,sm);return true;}
void VulkanTensor::destroy(){if(device_)VulkanMemoryAllocator::destroy(device_,gpuBuffer_,gpuMemory_);device_=VK_NULL_HANDLE;physical_=VK_NULL_HANDLE;queueFamily_=VK_QUEUE_FAMILY_IGNORED;shape_=tensor::TensorShape{};dtype_=tensor::TensorDType::Unknown;}
}
